import numpy as np
from scipy.spatial.distance import cdist
import gymnasium as gym
from gymnasium import spaces
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from typing import Tuple
from utils.geometry_utils import sat_collision_masked, check_bounds_torch, find_bl_placement, find_nfp_placement

# Assume sat_collision_masked, check_bounds_torch, and find_bl_placement are defined globally or imported

class NestingEnv(gym.Env):
    """
    Custom Environment for 2D Nesting Problem.
    Uses polygon data as NumPy arrays/tensors with masks.
    """
    # Modified __init__ to accept heuristic_type string and step_size
    def __init__(self, sheet_size=(400, 400), pieces_data=None, heuristic_type=None, heuristic_step_size=1.0, area_reward_scale=0.0001, distance_reward_scale=1.0, distance_threshold=20.0, distance_epsilon=1e-6, boundary_reward_scale=0.5):
        super(NestingEnv, self).__init__()

        self.sheet_size = sheet_size  # Size of the sheet (width, height)
        self.pieces_data = pieces_data if pieces_data is not None else []
        self.placed_pieces_coords = []  # List of placed piece coordinates (NumPy arrays/Tensors)
        self.placed_pieces_masks = [] # List of placed piece masks (NumPy arrays/Tensors)

        self.heuristic_type = heuristic_type # Store heuristic type string
        self.heuristic_step_size = heuristic_step_size # Store heuristic step size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Action space: [delta_x, delta_y, delta_rotation]
        # Adjusted action space bounds slightly for initial exploration
        self.action_space = spaces.Box(low=np.array([-30., -30., -360.]), high=np.array([100., 100., 360.]), dtype=np.float64)


        if self.pieces_data:
            single_piece_feature_dim = self.pieces_data[0]['features'].shape[0]
        else:
            single_piece_feature_dim = 6764 # Fallback

        # Calculate the size of the observation space
        # Base features + heuristic features (x, y, rotation)
        # Add 3 dimensions for heuristic placement (x, y, rotation)
        # If heuristic_type is None, these dimensions will be 0 or placeholder values
        heuristic_obs_dim = 3 # Assuming x, y, rotation are added
        total_observation_dim = single_piece_feature_dim + heuristic_obs_dim

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(total_observation_dim,), dtype=np.float64)

        if self.pieces_data:
            self.max_vertices = self.pieces_data[0]['polygon_coords'].shape[0]
        else:
            self.max_vertices = 10 # Fallback


        # Reward shaping parameters - ADJUSTED SCALES HERE
        self.area_reward_scale = area_reward_scale # Keep as is or adjust slightly if needed
        # Adjusted scales for proximity and boundary rewards
        self.distance_reward_scale = distance_reward_scale # Reduced from 1.0
        self.distance_threshold = distance_threshold # Keep the threshold
        self.distance_epsilon = distance_epsilon # Keep epsilon
        self.boundary_reward_scale = boundary_reward_scale # Reduced from 0.5


        # For rendering
        self.fig, self.ax = None, None
        # Store the last applied action for rendering the current piece
        self._last_action = np.array([0.0, 0.0, 0.0])
        # Store the last calculated heuristic placement for rendering
        self._last_heuristic_placement = np.array([-1.0, -1.0, 0.0]) # Store x, y, rotation (placeholder)


    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.placed_pieces_coords = []
        self.placed_pieces_masks = []
        self.current_piece_index = 0
        self.current_piece_data = self.pieces_data[self.current_piece_index] if self.pieces_data and self.current_piece_index < len(self.pieces_data) else None
        self.episode_finished = False # New flag to indicate if all pieces have been processed in this episode
        self._last_action = np.array([0.0, 0.0, 0.0]) # Reset last action
        self._last_heuristic_placement = np.array([-1.0, -1.0, 0.0]) # Reset last heuristic placement


        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        delta_x, delta_y, delta_rotation = action
        self._last_action = action # Store the current action for rendering

        reward = 0 # Initialize reward for this step
        terminated = False
        truncated = False

        if self.current_piece_data is None:
             # This should not happen if reset is called correctly and episode_finished is handled,
             # but as a safeguard:
             # Return zero observation, zero reward, terminated=True, truncated=False
             # Ensure the observation shape matches self.observation_space.shape
             return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype), 0, True, False, {"placed_pieces_count": len(self.placed_pieces_coords)}


        current_coords = self.current_piece_data['polygon_coords']
        current_mask = self.current_piece_data['polygon_mask']
        piece_features = self.current_piece_data['features']

        transformed_coords = current_coords.copy()

        # Apply translation
        transformed_coords[:, 0] += delta_x
        transformed_coords[:, 1] += delta_y

        # Apply rotation
        # Find the centroid of the actual vertices for rotation
        actual_current_coords = current_coords[current_mask]
        if actual_current_coords.shape[0] > 0:
            centroid = actual_current_coords.mean(axis=0)
        else:
            centroid = np.array([0.0, 0.0]) # Default centroid if no vertices


        angle_rad = np.deg2rad(delta_rotation)
        cos_theta = np.cos(angle_rad)
        sin_theta = np.sin(angle_rad)
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])

        # Rotate around the centroid
        transformed_coords_centered = transformed_coords - centroid
        rotated_coords_centered = transformed_coords_centered @ rotation_matrix
        transformed_piece_coords = rotated_coords_centered + centroid

        transformed_piece_mask = current_mask # Mask remains the same after transformation


        collided = self._check_collisions(transformed_piece_coords, transformed_piece_mask)
        out_of_bounds = self._check_bounds(transformed_piece_coords, transformed_piece_mask)

        # Calculate reward based on placement outcome
        reward = self._calculate_reward(transformed_piece_coords, transformed_piece_mask, collided, out_of_bounds, piece_features)


        if not collided and not out_of_bounds:
            # Successful placement
            self.placed_pieces_coords.append(transformed_piece_coords)
            self.placed_pieces_masks.append(transformed_piece_mask)
            # Reward was already calculated based on successful placement including area, proximity, boundary

        else:
            # Collision or out of bounds detected. Attempt heuristic placement.
            # Use the heuristic_type specified in __init__
            heuristic_x, heuristic_y = self._find_heuristic_placement(step_size=self.heuristic_step_size)
            print(f"colision or out of bounds detected, new heuristic location ({heuristic_x}, {heuristic_y})")
            
            # Check if heuristic found a valid placement (heuristic_x, heuristic_y not -1.0, -1.0)
            if heuristic_x > -1.0 or heuristic_y > -1.0:
                 # Heuristic found a placement. Place the piece at the heuristic location.
                 # We need to transform the piece to the heuristic location.
                 # This requires translating the original piece to the heuristic (x, y).
                 # Assuming the heuristic placement is the bottom-left corner of the piece's bounding box.
                 # Find the min x, min y of the actual vertices of the original piece.
                 original_current_coords = current_coords[current_mask]
                 if original_current_coords.shape[0] > 0:
                      min_piece_x = original_current_coords[:, 0].min()
                      min_piece_y = original_current_coords[:, 1].min()
            
                      # Calculate the translation needed from the piece's current position to the heuristic position
                      # The heuristic_x, heuristic_y are target coordinates, NOT deltas.
                      # We need to move the piece's min_x, min_y to heuristic_x, heuristic_y.
                      translation_x_heuristic = heuristic_x - min_piece_x
                      translation_y_heuristic = heuristic_y - min_piece_y
            
                      # Apply this translation to the original coordinates to get the heuristic placement coordinates
                      heuristic_placed_coords = original_current_coords + np.array([translation_x_heuristic, translation_y_heuristic])
            
                      # We should also consider rotation if the heuristic supports it.
                      # For now, assume heuristic_type 'bl' and 'nfp' only return x, y and no rotation.
                      # If heuristic_type provided rotation, we would apply it here around the piece's centroid.
                      # Let's assume no heuristic rotation for now.
            
                      # We need to store the padded coordinates with the mask for consistency with placed_pieces_coords
                      # Create a new padded array for the heuristic placement
                      heuristic_placed_padded_coords = np.zeros_like(current_coords)
                      heuristic_placed_padded_coords[current_mask] = heuristic_placed_coords # Place the transformed actual coords
                      # The mask remains the same
                      heuristic_placed_padded_mask = current_mask
            
            
                      # Check for collisions and bounds at the heuristic placement
                      heuristic_collided = self._check_collisions(heuristic_placed_padded_coords, heuristic_placed_padded_mask)
                      heuristic_out_of_bounds = self._check_bounds(heuristic_placed_padded_coords, heuristic_placed_padded_mask)
            
                      if not heuristic_collided and not heuristic_out_of_bounds:
                          print(f"heuristic not collided or not out of bounds")
                          # Heuristic placement is valid. Place the piece there.
                          self.placed_pieces_coords.append(heuristic_placed_padded_coords)
                          self.placed_pieces_masks.append(heuristic_placed_padded_mask)


        # Move to the next piece regardless of placement success
        self.current_piece_index += 1

        # Check if all pieces have been processed
        if self.current_piece_index >= len(self.pieces_data):
            terminated = True # Episode ends after attempting to place all pieces
            self.episode_finished = True # Set the flag
            self.current_piece_data = None # No more pieces to process
        else:
            # Load the next piece data if available
            self.current_piece_data = self.pieces_data[self.current_piece_index]


        observation = self._get_observation() # Get observation for the next step (next piece or empty)
        info = self._get_info()

        # Ensure info is updated even if observation is empty
        info["placed_pieces_count"] = len(self.placed_pieces_coords)


        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        if self.current_piece_data:
            # Get base features
            base_features = self.current_piece_data['features']

            # Calculate heuristic placement
            # Use the stored heuristic_type and heuristic_step_size
            # _find_heuristic_placement currently returns x, y
            heuristic_x, heuristic_y = self._find_heuristic_placement(step_size=self.heuristic_step_size)

            # Store the calculated heuristic placement for rendering
            # Assuming heuristic doesn't provide rotation, store 0.0 for rotation
            self._last_heuristic_placement = np.array([heuristic_x, heuristic_y, 0.0], dtype=np.float64)


            # Include heuristic observation in the observation vector
            if heuristic_x > -1.0 or heuristic_y > -1.0: # Check if a valid placement was potentially found (based on -1.0 indicator)
                # If a valid placement was found, use the calculated x, y and a default rotation observation
                 heuristic_obs = np.array([heuristic_x, heuristic_y, 0.0], dtype=np.float64) # Add 0.0 for rotation observation
            else:
                # If no valid heuristic placement found, use placeholder values
                heuristic_obs = np.array([-1.0, -1.0, -1.0], dtype=np.float64) # Use -1.0 as placeholder

            # Combine base features and heuristic observation
            # Ensure base_features is numpy array for concatenation
            if isinstance(base_features, torch.Tensor):
                 base_features_np = base_features.cpu().numpy()
            else:
                 base_features_np = base_features


            # Ensure the shapes are compatible before concatenating
            # If base_features_np is a scalar or has unexpected shape, handle it.
            # Assuming base_features_np is a 1D array/vector
            if base_features_np.ndim == 1:
                observation = np.concatenate([base_features_np, heuristic_obs])
            else:
                 # Handle unexpected shape, maybe return base features only or raise error
                 print(f"Warning: Unexpected shape for base_features_np: {base_features_np.shape}. Returning base features only.")
                 observation = base_features_np # Return base features if concatenation is not possible


            return observation
        else:
             # Return a zero observation with correct dimensionality if no more pieces
             # The zero observation must match the expanded observation space size
             total_observation_dim = self.observation_space.shape[0]
             self._last_heuristic_placement = np.array([-1.0, -1.0, 0.0]) # Reset heuristic placement on episode end
             return np.zeros(total_observation_dim, dtype=self.observation_space.dtype)


    def _get_info(self):
        return {"placed_pieces_count": len(self.placed_pieces_coords)}

    def _check_collisions(self, candidate_coords: np.ndarray, candidate_mask: np.ndarray):
        if not self.placed_pieces_coords:
            return False

        # Convert candidate to tensor and move to device
        candidate_tensor = torch.tensor(candidate_coords, dtype=torch.float32).unsqueeze(0).to(self.device)
        candidate_mask_tensor = torch.tensor(candidate_mask, dtype=torch.bool).unsqueeze(0).to(self.device)

        collision_detected = False
        # Iterate through each already placed piece
        for i in range(len(self.placed_pieces_coords)):
            placed_coords = self.placed_pieces_coords[i]
            placed_mask = self.placed_pieces_masks[i]

            # Convert placed piece to tensor and move to device
            # Ensure placed_coords and placed_mask are already tensors on the correct device from when they were added
            # If they were added as numpy arrays, convert them here
            if not isinstance(placed_coords, torch.Tensor):
                 placed_tensor = torch.tensor(placed_coords, dtype=torch.float32).unsqueeze(0).to(self.device)
                 placed_mask_tensor = torch.tensor(placed_mask, dtype=torch.bool).unsqueeze(0).to(self.device)
            else:
                 placed_tensor = placed_coords.unsqueeze(0).to(self.device) # Ensure batch dim is present
                 placed_mask_tensor = placed_mask.unsqueeze(0).to(self.device) # Ensure batch dim is present


            # Perform SAT collision check using the globally defined function
            # sat_collision_masked is expected to handle batch size 1 for these inputs
            # Ensure the input tensors to sat_collision_masked are on the same device
            # Directly call the global function
            collided = sat_collision_masked(
                candidate_tensor, candidate_mask_tensor, placed_tensor, placed_mask_tensor
            )

            if collided[0].item(): # collided is a boolean tensor of shape (1,)
                collision_detected = True
                break # No need to check against other placed pieces if a collision is found

        return collision_detected

    def _check_bounds(self, coords: np.ndarray, mask: np.ndarray):
        # Convert to tensor and move to device for the TorchScript function
        coords_tensor = torch.tensor(coords, dtype=torch.float32).to(self.device)
        mask_tensor = torch.tensor(mask, dtype=torch.bool).to(self.device)

        # Use the TorchScript check_bounds_torch function
        return check_bounds_torch(coords_tensor, mask_tensor, self.sheet_size)


    def _calculate_reward(self, piece_coords, piece_mask, collided, out_of_bounds, piece_features):
        reward = 0

        # Penalize for invalid placement
        if collided or out_of_bounds:
            # Reduce penalty compared to full episode termination
            # Consider a smaller negative reward or just 0 if no valid placement bonus is given
            reward = -0.9 # Example: small penalty for collision or out of bounds

        else:
            # Base reward for successful placement + area-based reward
            area_feature_index = 0
            # Ensure the index is valid for piece_features
            piece_area = piece_features[area_feature_index] if piece_features is not None and piece_features.shape[0] > area_feature_index else 0.0

            reward = 1.0 + piece_area * self.area_reward_scale # Reward for successfully placing the piece

            # --- Calculate minimum distance to placed pieces ---
            min_dist_to_placed = float('inf')
            actual_piece_coords = piece_coords[piece_mask]

            if self.placed_pieces_coords:
                # Only calculate distance if there are already placed pieces
                for placed_coords, placed_mask in zip(self.placed_pieces_coords, self.placed_pieces_masks):
                    actual_placed_coords = placed_coords[placed_mask]

                    # Ensure both current and placed pieces have actual vertices before calculating distance
                    if actual_piece_coords.shape[0] > 0 and actual_placed_coords.shape[0] > 0:
                        distance_matrix = cdist(actual_piece_coords, actual_placed_coords)
                        if distance_matrix.size > 0: # Ensure distance_matrix is not empty
                            min_dist_between_pair = distance_matrix.min()
                            min_dist_to_placed = min(min_dist_to_placed, min_dist_between_pair)

            # --- Add proximity reward based on min_dist_to_placed ---
            proximity_reward = 0.0
            # Reward for being close to other pieces (but not colliding)
            if min_dist_to_placed < float('inf') and min_dist_to_placed < self.distance_threshold:
                 # Proximity reward increases as distance decreases below the threshold
                 # Use a function that gives higher reward for smaller distances.
                 # Example: Inverse relationship or a decreasing exponential function.
                 # Simple inverse decay: reward = scale / (distance + epsilon)
                 proximity_reward = self.distance_reward_scale / (min_dist_to_placed + self.distance_epsilon)
                 # Cap the proximity reward to avoid extreme values for very small distances
                 proximity_reward = min(proximity_reward, self.distance_reward_scale * 5) # Cap at 5x the scale


            reward += proximity_reward
            # --- End proximity reward ---

            # --- Calculate minimum distance to sheet boundaries ---
            min_dist_to_sheet = float('inf')
            if actual_piece_coords.shape[0] > 0:
                dist_left = actual_piece_coords[:, 0].min()
                dist_right = self.sheet_size[0] - actual_piece_coords[:, 0].max()
                dist_bottom = actual_piece_coords[:, 1].min()
                dist_top = self.sheet_size[1] - actual_piece_coords[:, 1].max()
                min_dist_to_sheet = min(dist_left, dist_right, dist_bottom, dist_top)

            # --- Add boundary proximity reward based on min_dist_to_sheet ---
            boundary_proximity_reward = 0.0
            # Reward for being close to the boundary (but not out of bounds).
            # Similar logic to piece proximity reward.
            if min_dist_to_sheet < self.distance_threshold and min_dist_to_sheet >= 0: # Ensure distance is non-negative
                boundary_proximity_reward = self.boundary_reward_scale / (min_dist_to_sheet + self.distance_epsilon)
                 # Cap the boundary proximity reward
                boundary_proximity_reward = min(boundary_proximity_reward, self.boundary_reward_scale * 5) # Cap at 5x the scale

            reward += boundary_proximity_reward
            # --- End boundary proximity reward ---

        return reward

    # Modified _find_heuristic_placement to use self.heuristic_type string
    def _find_heuristic_placement(self, step_size: float = 1.0) -> Tuple[float, float]:
        """
        Finds a placement for the current piece using the specified heuristic type.
        Returns the found placement (x, y) or (-1.0, -1.0) if no placement found.
        """
        if self.current_piece_data is None:
            return -1.0, -1.0 # Cannot place if no current piece

        current_coords_np = self.current_piece_data['polygon_coords']
        current_mask_np = self.current_piece_data['polygon_mask']

        # Convert current piece to tensor and move to device, add batch dim (size 1)
        current_coords_tensor = torch.tensor(current_coords_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        current_mask_tensor = torch.tensor(current_mask_np, dtype=torch.bool).unsqueeze(0).to(self.device)

        # Convert placed pieces list of numpy arrays to list of tensors on device
        # This conversion should ideally happen when pieces are placed or passed in init,
        # but doing it here ensures tensors are on the correct device for the heuristic function.
        placed_pieces_coords_tensors = [
            torch.tensor(coords, dtype=torch.float32).to(self.device) for coords in self.placed_pieces_coords
        ]
        placed_pieces_masks_tensors = [
            torch.tensor(mask, dtype=torch.bool).to(self.device) for mask in self.placed_pieces_masks
        ]


        if self.heuristic_type == 'bl':
             # Call the TorchScript find_bl_placement function
             # find_bl_placement expects batch size 1 for the current piece
             # It expects lists of tensors for placed pieces
             x, y = find_bl_placement(
                 current_coords_tensor, current_mask_tensor,
                 placed_pieces_coords_tensors, placed_pieces_masks_tensors,
                 self.sheet_size, step_size=step_size
             )
             return x.item(), y.item() # Return as float
        elif self.heuristic_type == 'nfp':
             # Call the TorchScript find_nfp_placement function
             # find_nfp_placement expects batch size 1 for the current piece
             # It expects lists of tensors for placed pieces
             x, y = find_nfp_placement(
                 current_coords_tensor, current_mask_tensor,
                 placed_pieces_coords_tensors, placed_pieces_masks_tensors,
                 self.sheet_size, search_step_size=step_size # Using step_size for search_step_size
             )
             return x.item(), y.item() # Return as float

        else:
            # print(f"Warning: Unknown heuristic type '{self.heuristic_type}'. Returning invalid placement.") # Avoid printing in every step if heuristic_type is None
            return -1.0, -1.0 # Return -1.0, -1.0 if heuristic_type is None or unknown


    def render(self):
        if self.fig is None or self.ax is None:
            self.fig, self.ax = plt.subplots(1, 1, figsize=(self.sheet_size[0]/100, self.sheet_size[1]/100))
            self.ax.set_aspect('equal', adjustable='box')
            self.ax.set_xlim(0, self.sheet_size[0])
            self.ax.set_ylim(0, self.sheet_size[1])
            # Initial title without action
            self.ax.set_title("Nesting Environment")
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.grid(True)

        self.ax.clear()
        self.ax.set_aspect('equal', adjustable='box')
        self.ax.set_xlim(0, self.sheet_size[0])
        self.ax.set_ylim(0, self.sheet_size[1])

        # Get the last action for the title
        delta_x, delta_y, delta_rotation = self._last_action

        # Update the title to include placed pieces count and last action
        self.ax.set_title(f"Nesting Env (Placed: {len(self.placed_pieces_coords)} / Attempted: {self.current_piece_index})\nLast Action: dx={delta_x:.2f}, dy={delta_y:.2f}, dr={delta_rotation:.2f}")

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.grid(True)

        sheet_rect = MplPolygon([(0, 0), (self.sheet_size[0], 0), (self.sheet_size[0], self.sheet_size[1]), (0, self.sheet_size[1])],
                                closed=True, edgecolor='black', fill=False)
        self.ax.add_patch(sheet_rect)

        for piece_coords, piece_mask in zip(self.placed_pieces_coords, self.placed_pieces_masks):
            actual_coords = piece_coords[piece_mask]
            if actual_coords.shape[0] > 0:
                # Ensure the polygon is closed by adding the first vertex at the end
                if not np.array_equal(actual_coords[0], actual_coords[-1]):
                    plotting_coords = np.vstack([actual_coords, actual_coords[0]])
                else:
                    plotting_coords = actual_coords

                # Check if there are at least 3 unique points to form a polygon
                if np.unique(plotting_coords, axis=0).shape[0] >= 3:
                     mpl_polygon = MplPolygon(plotting_coords, closed=True, edgecolor='blue', facecolor='lightblue', alpha=0.7)
                     self.ax.add_patch(mpl_polygon)
                else:
                     # If not enough points for a polygon, plot as points
                     self.ax.plot(actual_coords[:, 0], actual_coords[:, 1], 'bo') # Blue circles for points


        # Render the current piece being attempted (if not all pieces are processed)
        if self.current_piece_data and self.current_piece_index < len(self.pieces_data):
             current_coords = self.current_piece_data['polygon_coords']
             current_mask = self.current_piece_data['polygon_mask']
             actual_current_coords = current_coords[current_mask]

             if actual_current_coords.shape[0] > 0:
                  # Use the stored last_action to transform the current piece for rendering
                  delta_x_action, delta_y_action, delta_rotation_action = self._last_action

                  # --- Render Piece at Agent's Action Location ---
                  temp_coords_action = current_coords.copy() # Start with original coords

                  # Apply translation
                  temp_coords_action[:, 0] += delta_x_action
                  temp_coords_action[:, 1] += delta_y_action

                   # Find the centroid of the actual vertices for rotation (using original coords)
                  original_current_coords = current_coords[current_mask]
                  if original_current_coords.shape[0] > 0:
                      centroid = original_current_coords.mean(axis=0)
                  else:
                       centroid = np.array([0.0, 0.0])

                  angle_rad_action = np.deg2rad(delta_rotation_action)
                  cos_theta_action = np.cos(angle_rad_action)
                  sin_theta_action = np.sin(angle_rad_action)
                  rotation_matrix_action = np.array([[cos_theta_action, -sin_theta_action],
                                                     [sin_theta_action, cos_theta_action]])

                  # Rotate around the centroid
                  temp_coords_centered_action = temp_coords_action - centroid
                  rotated_temp_coords_centered_action = temp_coords_centered_action @ rotation_matrix_action
                  transformed_current_coords_for_render_action = rotated_temp_coords_centered_action + centroid


                  # IMPORTANT: Filter out padded points and ensure the polygon is closed for plotting
                  actual_transformed_current_coords_action = transformed_current_coords_for_render_action[current_mask]

                  # Ensure the polygon is closed by adding the first vertex at the end
                  if actual_transformed_current_coords_action.shape[0] > 0 and not np.array_equal(actual_transformed_current_coords_action[0], actual_transformed_current_coords_action[-1]):
                      plotting_coords_current_action = np.vstack([actual_transformed_current_coords_action, actual_transformed_current_coords_action[0]])
                  else:
                      plotting_coords_current_action = actual_transformed_current_coords_action


                  # Check if there are at least 3 unique points to form a polygon
                  if np.unique(plotting_coords_current_action, axis=0).shape[0] >= 3:
                       mpl_polygon_current_action = MplPolygon(plotting_coords_current_action, closed=True, edgecolor='red', facecolor='salmon', alpha=0.7, label='Agent Placement') # Label for legend
                       self.ax.add_patch(mpl_polygon_current_action)
                  else:
                       # If not enough points, perhaps plot the actual points as markers
                       self.ax.plot(actual_transformed_current_coords_action[:, 0], actual_transformed_current_coords_action[:, 1], 'ro', label='Agent Placement Points') # Red circles for points, label for legend


                  # --- Render Piece at Heuristic Placement Location ---
                  heuristic_x, heuristic_y, heuristic_rotation = self._last_heuristic_placement # Get stored heuristic placement

                  # Only plot heuristic if a valid placement was found (not -1.0, -1.0)
                  if heuristic_x > -1.0 or heuristic_y > -1.0:
                       temp_coords_heuristic = current_coords.copy() # Start with original coords

                       # Apply translation based on heuristic placement
                       # Calculate the translation vector from the piece's min_x, min_y to the heuristic_x, heuristic_y
                       actual_current_coords = current_coords[current_mask]
                       if actual_current_coords.shape[0] > 0:
                            min_piece_x = actual_current_coords[:, 0].min()
                            min_piece_y = actual_current_coords[:, 1].min()
                            translation_x_heuristic = heuristic_x - min_piece_x
                            translation_y_heuristic = heuristic_y - min_piece_y
                       else:
                            translation_x_heuristic = heuristic_x # If no vertices, assume heuristic_x, heuristic_y is the target
                            translation_y_heuristic = heuristic_y


                       temp_coords_heuristic[:, 0] += translation_x_heuristic
                       temp_coords_heuristic[:, 1] += translation_y_heuristic

                       # Apply rotation based on heuristic rotation (assuming 0.0 for now)
                       # If heuristic_rotation were available:
                       # angle_rad_heuristic = np.deg2rad(heuristic_rotation)
                       # cos_theta_heuristic = np.cos(angle_rad_heuristic)
                       # sin_theta_heuristic = np.sin(angle_rad_heuristic)
                       # rotation_matrix_heuristic = np.array([[cos_theta_heuristic, -sin_theta_heuristic],
                       #                                      [sin_theta_heuristic, cos_theta_heuristic]])
                       # temp_coords_centered_heuristic = temp_coords_heuristic - centroid # Rotate around the same centroid
                       # rotated_temp_coords_centered_heuristic = temp_coords_centered_heuristic @ rotation_matrix_heuristic
                       # transformed_current_coords_for_render_heuristic = rotated_temp_coords_centered_heuristic + centroid
                       # For now, no heuristic rotation is applied here.

                       transformed_current_coords_for_render_heuristic = temp_coords_heuristic # Use translated coords if no heuristic rotation


                       # IMPORTANT: Filter out padded points and ensure the polygon is closed for plotting
                       actual_transformed_current_coords_heuristic = transformed_current_coords_for_render_heuristic[current_mask]


                       # Ensure the polygon is closed by adding the first vertex at the end
                       if actual_transformed_current_coords_heuristic.shape[0] > 0 and not np.array_equal(actual_transformed_current_coords_heuristic[0], actual_transformed_current_coords_heuristic[-1]):
                           plotting_coords_current_heuristic = np.vstack([actual_transformed_current_coords_heuristic, actual_transformed_current_coords_heuristic[0]])
                       else:
                           plotting_coords_current_heuristic = actual_transformed_current_coords_heuristic


                       # Check if there are at least 3 unique points to form a polygon
                       if np.unique(plotting_coords_current_heuristic, axis=0).shape[0] >= 3:
                            mpl_polygon_current_heuristic = MplPolygon(plotting_coords_current_heuristic, closed=True, edgecolor='green', facecolor='lightgreen', alpha=0.5, linestyle='--', label='Heuristic Placement') # Different color and linestyle
                            self.ax.add_patch(mpl_polygon_current_heuristic)
                       else:
                            # If not enough points, plot as points
                            self.ax.plot(actual_transformed_current_coords_heuristic[:, 0], actual_transformed_current_coords_heuristic[:, 1], 'go', linestyle='--', label='Heuristic Placement Points') # Green circles for points


        # Add a legend to distinguish between agent and heuristic placements
        if self.current_piece_data and self.current_piece_index < len(self.pieces_data):
            self.ax.legend()


        plt.draw()
        plt.pause(0.01)


    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig, self.ax = None, None