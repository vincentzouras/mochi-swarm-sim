# MPPI (Model Predictive Path Integral) for Blimp Position Control
import numpy as np
import torch
import torch.nn.functional as F

class BlimpMPPI:
    """
    MPPI controller for blimp position control.
    The dynamics model outputs velocity, which is integrated to get position.
    """
    def __init__(
        self,
        dynamics_model,
        horizon: int = 20,
        num_samples: int = 1000,
        lambda_: float = 1.0,
        sigma = 0.5,
        dt: float = 0.05,
        device: str = "cuda",
        pos_coeff: float = 1.0,
        vel_weight: float = 0.0,
        control_weight: float = 0.05,
        pos_weight = None,
    ):
        """
        Args:
            dynamics_model: The blimp dynamics model (sav_model) with .sample() method
            horizon: Prediction horizon (number of steps)
            num_samples: Number of action sequence samples
            lambda_: Temperature parameter for importance weighting
            sigma: Standard deviation for action noise (scalar or length-3 array for per-dimension)
            dt: Time step for integration
            device: Device to run computations on
            pos_coeff: Coefficient for position cost term (default: 1.0)
            vel_weight: Coefficient for velocity cost term (default: 0.0)
            control_weight: Coefficient for control effort cost term (default: 0.05)
            pos_weight: Per-dimension weights for position error [wx, wy, wz] (default: [1, 1, 1])
        """
        self.model = dynamics_model
        self.horizon = horizon
        self.num_samples = num_samples
        self.lambda_ = lambda_
        # Allow scalar sigma or per-dimension sigma (length-3 iterable)
        if isinstance(sigma, (int, float)):
            self.sigma = torch.full((3,), float(sigma), device=device, dtype=torch.float32)
        else:
            self.sigma = torch.as_tensor(sigma, device=device, dtype=torch.float32)
        self.dt = dt
        self.device = device
        self.pos_coeff = pos_coeff
        self.vel_weight = vel_weight
        self.control_weight = control_weight

        # Per-dimension weights for position error in cost (shape (3,))
        # Default: weight all position dimensions equally
        if pos_weight is None:
            self.pos_weight = torch.ones(3, device=device)
        else:
            self.pos_weight = torch.as_tensor(pos_weight, device=device, dtype=torch.float32)
        
        # Action bounds (adjust based on your action space)
        self.action_min = torch.tensor([0.0, 0.0, -np.pi + np.pi/2], device=device)
        self.action_max = torch.tensor([1.0, 1.0, np.pi + np.pi/2], device=device)
        
        # Initialize action sequence (will be updated each iteration)
        # Use the elementwise sum of min and max, broadcast over the horizon
        init_action = torch.tensor([0.0, 0.0, np.pi/2], device=device)  # shape (3,)
        self.action_sequence = init_action.unsqueeze(0).repeat(horizon, 1)
        
    def sample_action_sequences(self, mean_actions: torch.Tensor) -> torch.Tensor:
        """
        Sample action sequences from Gaussian distribution around mean.
        
        Args:
            mean_actions: Mean action sequence (horizon, 3)
            
        Returns:
            Sampled action sequences (num_samples, horizon, 3)
        """
        # noise_std has shape (1, 1, 3) so each action dimension can have its own sigma
        noise_std = self.sigma.view(1, 1, 3)
        noise = torch.randn(
            self.num_samples, self.horizon, 3,
            device=self.device
        ) * noise_std

        actions = mean_actions.unsqueeze(0) + noise  # (num_samples, horizon, 3)
        
        # Clip to action bounds
        actions = torch.clamp(actions, self.action_min, self.action_max)
        
        return actions
    
    def rollout_trajectories(
        self, 
        initial_state: torch.Tensor,
        initial_position: torch.Tensor,
        action_sequences: torch.Tensor
    ) -> tuple:
        """
        Roll out trajectories using the dynamics model.
        Integrates velocity to get position.
        Uses batched forward pass for efficiency.
        
        Args:
            initial_state: Initial state [v(3), w(3), rpy(3)] shape (9,)
            initial_position: Initial position [x, y, z] shape (3,)
            action_sequences: Action sequences (num_samples, horizon, 3)
            
        Returns:
            positions: (num_samples, horizon+1, 3) - includes initial position
            states: (num_samples, horizon+1, 9) - includes initial state
        """
        num_samples = action_sequences.shape[0]
        
        # Initialize trajectories
        positions = torch.zeros(
            (num_samples, self.horizon + 1, 3), 
            device=self.device
        )
        positions[:, 0] = initial_position.unsqueeze(0).repeat(num_samples, 1)
        
        states = torch.zeros(
            (num_samples, self.horizon + 1, 9), 
            device=self.device
        )
        states[:, 0] = initial_state.unsqueeze(0).repeat(num_samples, 1)
        
        # Initialize current observations for all samples (batched)
        current_obs = initial_state.unsqueeze(0).repeat(num_samples, 1)  # (num_samples, 9)
        
        # Rollout each step
        for t in range(self.horizon):
            # Get actions for this timestep (num_samples, 3)
            actions = action_sequences[:, t, :]  # (num_samples, 3)
            
            # Prepare batched input: [state, action] for all samples
            model_input = torch.cat([current_obs, actions], dim=-1)  # (num_samples, 12)
            model_input_norm = (model_input - self.model.x_mean) / self.model.x_std
            
            # Forward pass (batched) - much faster than sequential
            with torch.no_grad():
                next_obs_norm = self.model.forward(model_input_norm)  # (num_samples, 9)
                next_obs = next_obs_norm * self.model.y_std + self.model.y_mean
            
            states[:, t+1] = next_obs
            current_obs = next_obs  # Update for next iteration
            
            # Extract velocity (first 3 elements of state)
            velocities = next_obs[:, 0:3]  # (num_samples, 3)
            
            # Integrate velocity to get position
            positions[:, t+1] = positions[:, t] + velocities * self.dt
        
        return positions, states
    
    def compute_costs(
        self,
        positions: torch.Tensor,
        target_position: torch.Tensor,
        states: torch.Tensor = None,
        actions: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Compute cost for each trajectory.
        
        Args:
            positions: (num_samples, horizon+1, 3)
            target_position: (3,) target position [x, y, z]
            states: (num_samples, horizon+1, 9) optional, for additional costs
            
        Returns:
            costs: (num_samples,) total cost per trajectory
        """
        target = target_position.unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        
        # Position error cost (MSE over horizon), with per-dimension weights.
        # positions: (num_samples, horizon+1, 3)
        position_errors = positions - target
        weighted_sq_error = (position_errors ** 2) * self.pos_weight  # (num_samples, horizon+1, 3)
        position_costs = torch.sum(weighted_sq_error, dim=(1, 2))  # (num_samples,)

        # Optional: velocity penalty (e.g., for hover: penalize vertical speed)
        vel_costs = 0.0
        if states is not None and self.vel_weight > 0.0:
            # states: (num_samples, horizon+1, 9), linear velocities are states[..., 0:3]
            vel = states[..., 0:3]          # [vx, vy, vz]
            vz = vel[..., 2]                # vertical velocity (num_samples, horizon+1)
            vel_sq = vz ** 2
            # Sum over time dimension -> one scalar per trajectory
            vel_costs = torch.sum(vel_sq, dim=1)

        # Optional: control effort penalty
        control_costs = 0.0
        if actions is not None and self.control_weight > 0.0:
            control_costs = torch.sum(actions ** 2, dim=(1, 2))

        total_costs = self.pos_coeff * position_costs + self.vel_weight * vel_costs + self.control_weight * control_costs

        return total_costs
    
    def update_action_sequence(
        self,
        action_sequences: torch.Tensor,
        costs: torch.Tensor
    ) -> torch.Tensor:
        """
        Update action sequence using importance weighting.
        
        Args:
            action_sequences: (num_samples, horizon, 3)
            costs: (num_samples,)
            
        Returns:
            Updated mean action sequence (horizon, 3)
        """
        # Compute importance weights
        # w_i = exp(-(1/lambda) * (S_i - min(S)))
        min_cost = torch.min(costs)
        weights = torch.exp(-(1.0 / self.lambda_) * (costs - min_cost))
        weights = weights / (torch.sum(weights) + 1e-8)  # Normalize
        
        # Weighted average of action sequences
        weights_expanded = weights.unsqueeze(-1).unsqueeze(-1)  # (num_samples, 1, 1)
        updated_sequence = torch.sum(
            action_sequences * weights_expanded, 
            dim=0
        )  # (horizon, 3)
        
        return updated_sequence
    
    def compute_action(
        self,
        current_state: torch.Tensor,
        current_position: torch.Tensor,
        target_position: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute optimal action using MPPI.
        
        Args:
            current_state: Current state [v(3), w(3), rpy(3)] shape (9,)
            current_position: Current position [x, y, z] shape (3,)
            target_position: Target position [x, y, z] shape (3,)
            
        Returns:
            Optimal action (3,)
        """
        # Ensure tensors are on correct device
        if isinstance(current_state, np.ndarray):
            current_state = torch.from_numpy(current_state).float().to(self.device)
        if isinstance(current_position, np.ndarray):
            current_position = torch.from_numpy(current_position).float().to(self.device)
        if isinstance(target_position, np.ndarray):
            target_position = torch.from_numpy(target_position).float().to(self.device)
        
        # Sample action sequences around current mean
        action_sequences = self.sample_action_sequences(self.action_sequence)
        
        # Rollout trajectories
        positions, states = self.rollout_trajectories(
            current_state, current_position, action_sequences
        )
        
        # Compute costs
        costs = self.compute_costs(positions, target_position, states, action_sequences)
        
        # Update action sequence
        self.action_sequence = self.update_action_sequence(action_sequences, costs)
        
        # Return first action from updated sequence
        return self.action_sequence[0].clone()
    
    def reset(self):
        """Reset the action sequence."""
        self.action_sequence = torch.zeros((self.horizon, 3), device=self.device)