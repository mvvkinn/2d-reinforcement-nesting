import torch.nn as nn
import torch

class NestingDNN(nn.Module):
    def __init__(self, input_dim, action_dim=3):
        super(NestingDNN, self).__init__()
        # Define the common feature extraction layers
        # Simplified feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256), # Reduced from 1024
            nn.ReLU(),
            nn.Linear(256, 128), # Reduced from 512 -> 256
            nn.ReLU()
            # Removed layers: nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU()
        )

        # Actor head: predicts the action parameters (e.g., mean for continuous actions)
        # Input dimension should match the output of the simplified feature extractor (128)
        self.actor_head = nn.Linear(128, action_dim)

        # Critic head: predicts the value estimate of the current state
        # Input dimension should match the output of the simplified feature extractor (128)
        self.critic_head = nn.Linear(128, 1) # Output a single scalar value

        # Adding a parameter for log standard deviation for the Gaussian policy
        # Initialize it to a small value to encourage exploration
        # Initialize log_std to a slightly higher value to encourage more initial exploration
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0)) # Initializing log_std to -1.0 (std = exp(-1.0) approx 0.368) - adjusted from zeros


    def forward(self, x):
        """
        Performs the forward pass of the NestingDNN.

        Args:
            x (torch.Tensor): The input tensor containing the combined features.

        Returns:
            tuple: A tuple containing:
                - action_mean (torch.Tensor): The predicted mean of the action distribution.
                - value (torch.Tensor): The predicted value estimate of the state.
        """
        # Pass the input through the feature extractor
        features = self.feature_extractor(x)

        # Get the action mean from the actor head
        action_mean = self.actor_head(features)

        # Get the value estimate from the critic head
        value = self.critic_head(features)

        return action_mean, value

    def get_action_and_value(self, x, action=None):
        """
        Returns the action sampled from the policy and the value estimate.
        Also calculates log probabilities and entropy if an action is provided.
        """
        action_mean, value = self.forward(x)
        std = torch.exp(self.log_std)
        action_distribution = torch.distributions.Normal(action_mean, std)

        if action is None:
            # Sample action for interaction
            action = action_distribution.sample()

        # Calculate log probability of the action and entropy
        log_prob = action_distribution.log_prob(action).sum(dim=-1) # Sum over action dimensions
        entropy = action_distribution.entropy().sum(dim=-1) # Sum over action dimensions

        return action, log_prob, entropy, value

    def evaluate_action(self, x, action):
        """
        Evaluates the log probability and entropy of a given action under the current policy.
        Used during the PPO update phase.
        """
        action_mean, value = self.forward(x)
        std = torch.exp(self.log_std)
        action_distribution = torch.distributions.Normal(action_mean, std)

        log_prob = action_distribution.log_prob(action).sum(dim=-1)
        entropy = action_distribution.entropy().sum(dim=-1)

        return log_prob, entropy, value