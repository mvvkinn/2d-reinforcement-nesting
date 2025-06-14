import torch
from typing import Tuple, List

# --- Normalize helper ---
@torch.jit.script
def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # Handle zero vectors that might result from padded edges
    norm = v.norm(p=2, dim=-1, keepdim=True)
    # Avoid division by zero for zero vectors
    # Use a small epsilon for comparison with norm
    return v / (norm + eps) if torch.any(norm > eps) else torch.zeros_like(v) # Use torch.any for batch dim

# 성능개선판
@torch.jit.script
def get_axes(p: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
    edges = p.roll(-1, dims=1) - p
    rolled_mask = mask.roll(-1, dims=1)
    valid_edges_mask_bool = mask & rolled_mask
    potential_axes = torch.stack([-edges[:, :, 1], edges[:, :, 0]], dim=-1)

    # Normalize only valid axes
    flat_axes = potential_axes.view(-1, 2)
    flat_mask = valid_edges_mask_bool.view(-1)
    flat_normalized = torch.zeros_like(flat_axes)
    if torch.any(flat_mask):
        flat_normalized[flat_mask] = normalize(flat_axes[flat_mask])

    normalized_axes = flat_normalized.view_as(potential_axes)
    return [normalized_axes, valid_edges_mask_bool]


@torch.jit.script
def batch_proj(p: torch.Tensor, mask: torch.Tensor, axis: torch.Tensor) -> List[torch.Tensor]:
    # p: (B, N, 2), axis: (2,)
    projections = torch.einsum('bnd,d->bn', p, axis)
    masked_proj = torch.where(mask, projections, torch.tensor(float('-inf'), device=p.device))

    min_proj = masked_proj.clone()
    min_proj[~mask] = float('inf')
    min_val = min_proj.min(dim=1).values
    max_val = masked_proj.max(dim=1).values

    return [min_val, max_val]


@torch.jit.script
def batch_proj_general(p: torch.Tensor, mask: torch.Tensor, axis: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N, _ = p.shape
    A = axis.shape[0]

    p_exp = p.unsqueeze(1).expand(B, A, N, 2)
    axis_exp = axis.unsqueeze(0).unsqueeze(2).expand(B, A, N, 2)

    proj = torch.einsum('band,band->ban', p_exp, axis_exp)

    mask_exp = mask.unsqueeze(1).expand(B, A, N)
    proj = torch.where(mask_exp, proj, torch.full_like(proj, float('-inf')))
    proj_min = proj.clone()
    proj_min[~mask_exp] = float('inf')

    return proj_min.min(dim=2).values, proj.max(dim=2).values


@torch.jit.script
def sat_collision_masked(
    p1: torch.Tensor, mask1: torch.Tensor,
    p2: torch.Tensor, mask2: torch.Tensor
) -> List[torch.Tensor]:
    B, N_max, _ = p1.shape

    # 1. 축 계산
    axes1, axes_mask1 = get_axes(p1, mask1)
    axes2, axes_mask2 = get_axes(p2, mask2)
    all_axes = torch.cat([axes1, axes2], dim=1)
    all_axes_mask = torch.cat([axes_mask1, axes_mask2], dim=1)

    # 2. 축 마스킹
    flat_axes = all_axes.view(-1, 2)
    flat_axes_mask = all_axes_mask.view(-1)
    valid_axes = flat_axes[flat_axes_mask]  # (?, 2)

    if valid_axes.size(0) == 0:
        is_empty1 = ~mask1.any(dim=1)
        is_empty2 = ~mask2.any(dim=1)
        empty_both = is_empty1 & is_empty2
        collision = ~(is_empty1 | is_empty2) & ~empty_both
        return [collision]

    # 3. projection (einsum 적용)
    p1_min, p1_max = batch_proj_general(p1, mask1, valid_axes)
    p2_min, p2_max = batch_proj_general(p2, mask2, valid_axes)

    # 4. SAT 조건 확인
    eps = 1e-6
    no_overlap = (p1_max + eps < p2_min) | (p2_max + eps < p1_min)
    any_gap = no_overlap.any(dim=1)
    collision = ~any_gap

    return [collision]

@torch.jit.script
def sat_collision_masked_batch(
    p1: torch.Tensor,      # (B, N, 2)
    mask1: torch.Tensor,   # (B, N)
    p2: torch.Tensor,      # (B, M, 2)
    mask2: torch.Tensor    # (B, M)
) -> torch.Tensor:         # (B,) bool - True if collision occurs
    def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norm = v.norm(p=2, dim=-1, keepdim=True)
        return v / (norm + eps)

    B, N, _ = p1.shape
    _, M, _ = p2.shape

    # 축 계산 (법선 벡터)
    p1_edges = p1.roll(shifts=-1, dims=1) - p1
    p1_axes = torch.stack([-p1_edges[..., 1], p1_edges[..., 0]], dim=-1)
    p1_axes = normalize(p1_axes)

    p2_edges = p2.roll(shifts=-1, dims=1) - p2
    p2_axes = torch.stack([-p2_edges[..., 1], p2_edges[..., 0]], dim=-1)
    p2_axes = normalize(p2_axes)

    # 모든 축 병합 (B, A, 2)
    all_axes = torch.cat([p1_axes, p2_axes], dim=1)           # (B, A, 2)
    all_mask = torch.cat([mask1, mask2], dim=1)               # (B, A)

    A = all_axes.size(1)

    # --- Projection ---
    # p1_proj: (B, A, N), 각 축에 대해 모든 점을 투영
    p1_proj = torch.einsum("ban,bnj->ban", all_axes, p1)      # (B, A, N)
    p2_proj = torch.einsum("ban,bmj->bam", all_axes, p2)      # (B, A, M)

    # 마스킹 적용
    inf = float("inf")
    ninf = float("-inf")

    p1_proj_masked_min = p1_proj.masked_fill(~mask1.unsqueeze(1), inf).amin(dim=2)  # (B, A)
    p1_proj_masked_max = p1_proj.masked_fill(~mask1.unsqueeze(1), ninf).amax(dim=2)

    p2_proj_masked_min = p2_proj.masked_fill(~mask2.unsqueeze(1), inf).amin(dim=2)
    p2_proj_masked_max = p2_proj.masked_fill(~mask2.unsqueeze(1), ninf).amax(dim=2)

    # 오버랩 판단 (B, A)
    overlap = (p1_proj_masked_max >= p2_proj_masked_min) & (p2_proj_masked_max >= p1_proj_masked_min)

    # 하나라도 분리축 존재 -> 충돌 안함
    any_gap = ~overlap.any(dim=1)  # (B,) - True if separating axis found
    return ~any_gap  # True if collision occurred


@torch.jit.script
def support_masked(p: torch.Tensor, mask: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
    # d: (B, 2), p: (B, N, 2)
    proj = torch.einsum("bnd,bd->bn", p, d)  # (B, N)
    masked_proj = torch.where(mask, proj, torch.tensor(float("-inf"), device=p.device))
    max_idx = masked_proj.argmax(dim=1, keepdim=True)  # (B, 1)
    support = torch.gather(p, 1, max_idx.unsqueeze(-1).expand(-1, 1, 2)).squeeze(1)  # (B, 2)
    return support


@torch.jit.script
def sat_gjk_parallel(
    candidates: torch.Tensor,        # (K, N, 2)
    candidates_mask: torch.Tensor,   # (K, N)
    placed: torch.Tensor,            # (M, N, 2)
    placed_mask: torch.Tensor        # (M, N)
) -> Tuple[torch.Tensor, torch.Tensor]:
    K, N, _ = candidates.shape
    M, _, _ = placed.shape

    # broadcast
    c_exp = candidates.unsqueeze(1).expand(K, M, N, 2)
    c_mask_exp = candidates_mask.unsqueeze(1).expand(K, M, N)
    p_exp = placed.unsqueeze(0).expand(K, M, N, 2)
    p_mask_exp = placed_mask.unsqueeze(0).expand(K, M, N)

    # flatten
    c_flat = c_exp.reshape(-1, N, 2)
    c_mask_flat = c_mask_exp.reshape(-1, N)
    p_flat = p_exp.reshape(-1, N, 2)
    p_mask_flat = p_mask_exp.reshape(-1, N)

    collisions_flat = sat_collision_masked(c_flat, c_mask_flat, p_flat, p_mask_flat)[0]
    collision_matrix = collisions_flat.view(K, M)
    dummy_dist = torch.full((K, M), float('inf'), device=candidates.device)

    return collision_matrix, dummy_dist


# Redefine check_bounds_torch
@torch.jit.script
def check_bounds_torch(
    piece_coords: torch.Tensor, # (N_max, 2)
    piece_mask: torch.Tensor,   # (N_max,)
    sheet_size: Tuple[float, float]
) -> bool:
    sheet_width = torch.tensor(sheet_size[0], dtype=piece_coords.dtype, device=piece_coords.device)
    sheet_height = torch.tensor(sheet_size[1], dtype=piece_coords.dtype, device=piece_coords.device)

    actual_coords = piece_coords[piece_mask]

    if actual_coords.numel() == 0:
        return False # An empty piece is not out of bounds

    min_x = torch.min(actual_coords[:, 0])
    max_x = torch.max(actual_coords[:, 0])
    min_y = torch.min(actual_coords[:, 1])
    max_y = torch.max(actual_coords[:, 1])

    is_out = (min_x < -1e-6) or (max_x > sheet_width + 1e-6) or (min_y < -1e-6) or (max_y > sheet_height + 1e-6)

    return is_out

@torch.jit.script
def check_bounds_torch_batch(
    coords: torch.Tensor,     # (B, N, 2)
    mask: torch.Tensor,       # (B, N)
    sheet_size: Tuple[float, float]
) -> torch.Tensor:            # (B,) bool
    sheet_w = sheet_size[0]
    sheet_h = sheet_size[1]

    x = coords[..., 0]  # (B, N)
    y = coords[..., 1]  # (B, N)

    x_in = (x >= 0.0) & (x <= sheet_w)
    y_in = (y >= 0.0) & (y <= sheet_h)

    inside = x_in & y_in & mask  # (B, N) 유효 좌표만 검사
    valid = inside.sum(dim=1) == mask.sum(dim=1)  # 모든 유효좌표가 안에 있으면 OK
    return ~valid  # True: 범위 벗어남


@torch.jit.script
def calculate_minkowski_difference(
    p1_coords: torch.Tensor,  # (B, N_max, 2) padded coordinates of polygon 1
    p1_mask: torch.Tensor,    # (B, N_max) boolean mask for polygon 1
    p2_coords: torch.Tensor,  # (B, M_max, 2) padded coordinates of polygon 2
    p2_mask: torch.Tensor     # (B, M_max) boolean mask for polygon 2
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculates the vertices of the Minkowski Difference (p1 - p2) between two polygons
    using pure PyTorch tensor operations.
    The Minkowski Difference vertices are calculated as the difference of all pairs
    of vertices between p1 and p2.
    Args:
        p1_coords (torch.Tensor): (B, N_max, 2) padded coordinates of polygon 1.
        p1_mask (torch.Tensor): (B, N_max) boolean mask for polygon 1.
        p2_coords (torch.Tensor): (B, M_max, 2) padded coordinates of polygon 2.
        p2_mask (torch.Tensor): (B, M_max) boolean mask for polygon 2.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - md_vertices (torch.Tensor): (B, N_max * M_max, 2) padded coordinates
                                          of the Minkowski Difference polygon vertices.
            - md_mask (torch.Tensor): (B, N_max * M_max) boolean mask for
                                      the Minkowski Difference vertices.
    """
    B, N_max, _ = p1_coords.shape
    _, M_max, _ = p2_coords.shape

    # Initialize the resulting vertices and mask with padding
    # The maximum possible number of vertices in the Minkowski Difference
    # is the product of the number of vertices in the input polygons.
    max_md_vertices = N_max * M_max
    md_vertices = torch.zeros((B, max_md_vertices, 2), dtype=p1_coords.dtype, device=p1_coords.device)
    md_mask = torch.zeros((B, max_md_vertices), dtype=torch.bool, device=p1_coords.device)

    # Iterate through each batch element
    for b in range(B):
        # Get actual vertices and masks for the current batch element
        actual_p1_coords_b = p1_coords[b][p1_mask[b]] # (num_actual_p1_b, 2)
        actual_p2_coords_b = p2_coords[b][p2_mask[b]] # (num_actual_p2_b, 2)

        num_actual_p1_b = actual_p1_coords_b.size(0)
        num_actual_p2_b = actual_p2_coords_b.size(0)

        if num_actual_p1_b == 0 or num_actual_p2_b == 0:
             # If either polygon is empty (no actual vertices), the Minkowski Difference is empty.
             # The initialized md_vertices and md_mask (all zeros/false) are correct in this case.
             continue

        # Compute the difference of all pairs of actual vertices
        # Reshape actual_p1_coords_b to (num_actual_p1_b, 1, 2)
        # Reshape actual_p2_coords_b to (1, num_actual_p2_b, 2)
        # Subtracting them broadcast: (num_actual_p1_b, num_actual_p2_b, 2)
        # Then flatten the result to (num_actual_p1_b * num_actual_p2_b, 2)
        pair_differences_b = actual_p1_coords_b.unsqueeze(1) - actual_p2_coords_b.unsqueeze(0)
        flattened_differences_b = pair_differences_b.view(-1, 2) # (num_actual_p1_b * num_actual_p2_b, 2)

        # Store the computed vertices in the padded tensor
        num_computed_vertices_b = flattened_differences_b.size(0)
        md_vertices[b, :num_computed_vertices_b] = flattened_differences_b

        # Update the mask for the actual vertices in this batch element
        md_mask[b, :num_computed_vertices_b] = True

    return md_vertices, md_mask


@torch.jit.script
def calculate_minkowski_difference(
    p1_coords: torch.Tensor,  # (B, N_max, 2)
    p1_mask: torch.Tensor,    # (B, N_max)
    p2_coords: torch.Tensor,  # (B, M_max, 2)
    p2_mask: torch.Tensor     # (B, M_max)
) -> Tuple[torch.Tensor, torch.Tensor]:
    B, N_max, _ = p1_coords.shape
    _, M_max, _ = p2_coords.shape
    max_md_vertices = N_max * M_max

    # 마스킹 적용: 0으로 padding된 좌표는 0 벡터로 유지
    p1_mask_f = p1_mask.unsqueeze(-1).float()  # (B, N_max, 1)
    p2_mask_f = p2_mask.unsqueeze(-1).float()  # (B, M_max, 1)

    p1_masked = p1_coords * p1_mask_f  # (B, N_max, 2)
    p2_masked = p2_coords * p2_mask_f  # (B, M_max, 2)

    # 브로드캐스팅 기반 차이 계산: (B, N_max, M_max, 2)
    p1_exp = p1_masked.unsqueeze(2)  # (B, N_max, 1, 2)
    p2_exp = p2_masked.unsqueeze(1)  # (B, 1, M_max, 2)
    diffs = p1_exp - p2_exp          # (B, N_max, M_max, 2)

    # 마스크 계산: (B, N_max, M_max)
    valid_mask = p1_mask.unsqueeze(2) & p2_mask.unsqueeze(1)  # bool mask
    valid_mask_flat = valid_mask.view(B, -1)                  # (B, N_max*M_max)
    diffs_flat = diffs.view(B, -1, 2)                         # (B, N_max*M_max, 2)

    return diffs_flat, valid_mask_flat


# Redefine find_nfp_placement to fix the tensor creation issue
@torch.jit.script
def find_nfp_placement(
    current_piece_coords: torch.Tensor, # (B, N_max, 2)
    current_piece_mask: torch.Tensor,   # (B, N_max)
    placed_pieces_coords_list: List[torch.Tensor], # List of (M_i, 2) tensors
    placed_pieces_masks_list: List[torch.Tensor],   # List of (M_i,) tensors
    sheet_size: Tuple[float, float],
    search_step_size: float # Step size for grid search
) -> Tuple[torch.Tensor, torch.Tensor]: # Return x, y coordinates
    device = current_piece_coords.device
    dtype = current_piece_coords.dtype
    B, N_max, _ = current_piece_coords.shape

    # print(f"find_nfp_placement - current_piece_coords device: {current_piece_coords.device}") # Debug print

    if B != 1:
        return torch.tensor(-1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device)

    current_piece_coords_b = current_piece_coords.squeeze(0) # (N_max, 2)
    current_piece_mask_b = current_piece_mask.squeeze(0)     # (N_max,)


    actual_current_coords = current_piece_coords_b[current_piece_mask_b]
    if actual_current_coords.numel() == 0:
        return torch.tensor(-1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device)

    min_piece_x = torch.min(actual_current_coords[:, 0])
    min_piece_y = torch.min(actual_current_coords[:, 1])
    max_piece_x = torch.max(actual_current_coords[:, 0])
    max_piece_y = torch.max(actual_current_coords[:, 1])
    piece_width = max_piece_x - min_piece_x
    piece_height = max_piece_y - min_piece_y

    sheet_width_tensor = torch.tensor(sheet_size[0], dtype=dtype, device=device)
    sheet_height_tensor = torch.tensor(sheet_size[1], dtype=dtype, device=device)

    max_search_x = sheet_width_tensor - piece_width
    max_search_y = sheet_height_tensor - piece_height

    # Add check for piece larger than sheet
    if max_search_x < -1e-6 or max_search_y < -1e-6:
         return torch.tensor(-1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device)


    search_x_coords = torch.arange(0.0, max_search_x.item() + search_step_size/2, search_step_size, dtype=dtype, device=device)
    search_y_coords = torch.arange(0.0, max_search_y.item() + search_step_size/2, search_step_size, dtype=dtype, device=device)

    if search_x_coords.numel() == 0 or search_y_coords.numel() == 0:
         return torch.tensor(-1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device)


    grid_x, grid_y = torch.meshgrid(search_x_coords, search_y_coords, indexing='ij')
    potential_translations = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)

    best_x = torch.tensor(-1.0, dtype=dtype, device=device)
    best_y = torch.tensor(-1.0, dtype=dtype, device=device)
    found_placement = False

    large_number = sheet_width_tensor * 2.0
    sort_keys = potential_translations[:, 1] * large_number + potential_translations[:, 0] # Sort bottom-left first
    sorted_indices = torch.argsort(sort_keys)

    sorted_potential_translations = potential_translations[sorted_indices]


    for i in range(sorted_potential_translations.size(0)):
        translation = sorted_potential_translations[i]

        # Fix: Perform tensor subtraction directly
        actual_translation_vector = torch.stack([translation[0] - min_piece_x, translation[1] - min_piece_y])

        temp_piece_coords = current_piece_coords_b + actual_translation_vector.unsqueeze(0)
        temp_piece_mask = current_piece_mask_b

        collided_with_placed = False
        if placed_pieces_coords_list:
             for placed_coords, placed_mask in zip(placed_pieces_coords_list, placed_pieces_masks_list):
                 placed_coords_dev = placed_coords.to(device=device, dtype=dtype)
                 placed_mask_dev = placed_mask.to(device=device)
                 # print(f"find_nfp_placement - placed_coords_dev device: {placed_coords_dev.device}") # Debug print

                 collided_list = sat_collision_masked(
                     temp_piece_coords.unsqueeze(0), temp_piece_mask.unsqueeze(0),
                     placed_coords_dev.unsqueeze(0), placed_mask_dev.unsqueeze(0)
                 )
                 collided = collided_list[0]

                 if collided[0]:
                     collided_with_placed = True
                     break

        out_of_bounds = check_bounds_torch(temp_piece_coords, temp_piece_mask, sheet_size)


        if not collided_with_placed and not out_of_bounds:
            best_x = translation[0]
            best_y = translation[1]
            found_placement = True
            break

    return best_x, best_y


@torch.jit.script
def find_bl_placement(
    current_piece_coords: torch.Tensor, # (B, N_max, 2)
    current_piece_mask: torch.Tensor,   # (B, N_max)
    placed_pieces_coords_list: List[torch.Tensor], # List of (M_i, 2) tensors
    placed_pieces_masks_list: List[torch.Tensor],   # List of (M_i,) tensors
    sheet_size: Tuple[float, float],
    step_size: float = 1.0, # Step size for scanning
    vertical_step_multiplier: float = 2.0 # How much to step up vertically
) -> Tuple[torch.Tensor, torch.Tensor]: # Return x, y coordinates
    device = current_piece_coords.device
    dtype = current_piece_coords.dtype
    B, N_max, _ = current_piece_coords.shape

    # print(f"find_bl_placement - current_piece_coords device: {current_piece_coords.device}") # Debug print

    # This BL implementation is designed for a single piece (batch size 1)
    if B != 1:
        # Return invalid placement for batch size > 1
        return torch.tensor(-1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device)

    current_piece_coords_b = current_piece_coords.squeeze(0) # (N_max, 2)
    current_piece_mask_b = current_piece_mask.squeeze(0)     # (N_max,)

    actual_current_coords = current_piece_coords_b[current_piece_mask_b]
    if actual_current_coords.numel() == 0:
        # Cannot place an empty piece
        return torch.tensor(-1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device)

    min_piece_x = torch.min(actual_current_coords[:, 0])
    min_piece_y = torch.min(actual_current_coords[:, 1])
    max_piece_x = torch.max(actual_current_coords[:, 0])
    max_piece_y = torch.max(actual_current_coords[:, 1])
    piece_width = max_piece_x - min_piece_x
    piece_height = max_piece_y - min_piece_y

    sheet_width_tensor = torch.tensor(sheet_size[0], dtype=dtype, device=device)
    sheet_height_tensor = torch.tensor(sheet_size[1], dtype=dtype, device=device)

    # Adjust scan range based on piece size
    scan_x_range = sheet_width_tensor - piece_width
    scan_y_range = sheet_height_tensor - piece_height

    # Add check for piece larger than sheet
    if scan_x_range < -1e-6 or scan_y_range < -1e-6:
         return torch.tensor(-1.0, dtype=dtype, device=device), torch.tensor(-1.0, dtype=dtype, device=device)


    # Start scanning from the bottom-left
    best_x = torch.tensor(-1.0, dtype=dtype, device=device)
    best_y = torch.tensor(-1.0, dtype=dtype, device=device)
    found_placement = False

    # Iterate through potential y positions (rows)
    y_scan_start = 0.0
    # Ensure the upper bound for y_steps is not less than the lower bound
    y_scan_end = scan_y_range + step_size/2
    # Check if y_scan_end is less than y_scan_start before creating the range
    if y_scan_end < y_scan_start:
         y_steps = torch.tensor([], dtype=dtype, device=device) # Create an empty tensor if the range is invalid
    else:
        y_steps = torch.arange(y_scan_start, y_scan_end, step_size * vertical_step_multiplier, dtype=dtype, device=device)


    for current_y in y_steps:
        # Iterate through potential x positions (columns) for the current y
        x_scan_start = 0.0
        # Ensure x_scan_start doesn't exceed max_search_x (sheet_width - piece_width)
        x_scan_end = scan_x_range + step_size/2
        # Check if x_scan_end is less than x_scan_start before creating the range
        if x_scan_end < x_scan_start:
             x_steps = torch.tensor([], dtype=dtype, device=device) # Create an empty tensor if the range is invalid
        else:
            x_steps = torch.arange(x_scan_start, x_scan_end, step_size, dtype=dtype, device=device)

        for current_x in x_steps:
            # Calculate the actual translation needed to place the piece's min_x, min_y at (current_x, current_y)
            # Fix: Perform tensor subtraction directly
            actual_translation_vector = torch.stack([current_x - min_piece_x, current_y - min_piece_y])

            # Apply the potential translation
            temp_piece_coords = current_piece_coords_b + actual_translation_vector.unsqueeze(0)
            temp_piece_mask = current_piece_mask_b

            # Check for collision with placed pieces
            collided_with_placed = False
            if placed_pieces_coords_list:
                # Convert placed pieces to tensors and move to device if they aren't already
                # (Assuming they are already on device and correct dtype from NestingEnv)
                for placed_coords, placed_mask in zip(placed_pieces_coords_list, placed_pieces_masks_list):
                     # Ensure placed pieces are on the correct device and dtype
                     placed_coords_dev = placed_coords.to(device=device, dtype=dtype)
                     placed_mask_dev = placed_mask.to(device=device)
                     # print(f"find_bl_placement - placed_coords_dev device: {placed_coords_dev.device}") # Debug print


                     # Use the masked SAT collision check
                     # sat_collision_masked expects batch dimension, so unsqueeze
                     collided_list = sat_collision_masked(
                        temp_piece_coords.unsqueeze(0), temp_piece_mask.unsqueeze(0),
                        placed_coords_dev.unsqueeze(0), placed_mask_dev.unsqueeze(0)
                    )
                     collided = collided_list[0] # Get the boolean tensor result

                     if collided.item(): # Check the boolean value
                         collided_with_placed = True
                         break # No need to check against other placed pieces if a collision is found

            # Check if the current piece is within the sheet bounds after translation
            # check_bounds_torch expects (N_max, 2) and (N_max,) inputs for a single piece
            out_of_bounds = check_bounds_torch(temp_piece_coords, temp_piece_mask, sheet_size)


            # If no collision and within bounds, this is a valid BL placement
            if not collided_with_placed and not out_of_bounds:
                best_x = current_x
                best_y = current_y
                found_placement = True
                break # Found a placement, exit inner loop
        if found_placement:
            break # Found a placement, exit outer loop

    return best_x, best_y # Return the found bottom-left placement coordinates or (-1.0, -1.0)