# dynamics_nn.py
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

class DiscreteQuadDynamicsNN(nn.Module):
    """
    Feed-forward NN for discrete-time dynamics with our data format:
      input  x = [state(9), action(3)] -> shape (B, 12)
      output y = next_state (9)        -> shape (B, 9)
    Hidden layers: 128 -> 64 -> 64 with ELU activations.
    """
    def __init__(self, in_dim: int = 12, hidden=(128, 64, 64), out_dim: int = 9):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # print(self.device)
        h1, h2, h3 = hidden
        self.fc1 = nn.Linear(in_dim, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, h3)
        self.out = nn.Linear(h3, out_dim)  # <— last layer (adapt online)
        self.activation = nn.ELU()
        # self.x_mean = None
        # self.x_std = None
        # self.y_mean = None
        # self.y_std = None
        self.register_buffer("x_mean", torch.empty(in_dim))
        self.register_buffer("x_std",  torch.empty(in_dim))
        self.register_buffer("y_mean", torch.empty(out_dim))
        self.register_buffer("y_std",  torch.empty(out_dim))
        # self.double()

    def reset(self, initial_obs_batch: np.ndarray, return_as_np: bool = True):
        # initial_obs_batch = torch.from_numpy(initial_obs_batch).float().to(self.device)
        model_state = {"obs": initial_obs_batch}
        return model_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.activation(self.fc1(x))
        z = self.activation(self.fc2(z))
        z = self.activation(self.fc3(z))
        y = self.out(z)  # (B, 10)
        return y

    @staticmethod
    def split_outputs(y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split network output into (v_next, omega_next, q_next_raw).
        Shapes: (B,3), (B,3), (B,4)
        """
        v_next   = y[:, 0:3]
        w_next   = y[:, 3:6]
        q_next_r = y[:, 6:10]
        return v_next, w_next, q_next_r

    def split_outputs_rpy(self, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Split network output into (v_next, omega_next, q_next_raw).
        Shapes: (B,3), (B,3), (B,4)
        """
        v_next   = y[:, 0:3]
        w_next   = y[:, 3:6]
        rpy_next = y[:, 6:9]
        return v_next, w_next, rpy_next
    # @staticmethod
    # def quat_to_R_nonunit(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    #     """
    #     Map (possibly non-unit) quaternion(s) to rotation matrix/matrices via:
    #         R = Q / ||q||^2
    #     where Q is the standard 3x3 from quaternion components.
    #     q: (B,4) with components [qw, qx, qy, qz]
    #     returns R: (B,3,3)
    #     """
    #     qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    #     qw2, qx2, qy2, qz2 = qw*qw, qx*qx, qy*qy, qz*qz

    #     # Elements of Q
    #     r00 = qw2 + qx2 - qy2 - qz2
    #     r01 = 2*(qx*qy - qw*qz)
    #     r02 = 2*(qx*qz + qw*qy)

    #     r10 = 2*(qx*qy + qw*qz)
    #     r11 = qw2 + qy2 - qx2 - qz2
    #     r12 = 2*(qy*qz - qw*qx)

    #     r20 = 2*(qx*qz - qw*qy)
    #     r21 = 2*(qy*qz + qw*qx)
    #     r22 = qw2 + qz2 - qx2 - qy2

    #     Qmat = torch.stack([
    #         torch.stack([r00, r01, r02], dim=-1),
    #         torch.stack([r10, r11, r12], dim=-1),
    #         torch.stack([r20, r21, r22], dim=-1),
    #     ], dim=-2)  # (B,3,3)

    #     norm2 = (qw2 + qx2 + qy2 + qz2).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)  # (B,1,1)
    #     R = Qmat / norm2
    #     return R

    # ---------- Utilities for online adaptation (last-layer-only) ----------
    def freeze_all_but_last(self):
        for p in self.parameters():
            p.requires_grad_(False)
        for p in self.out.parameters():
            p.requires_grad_(True)

    def unfreeze_all(self):
        for p in self.parameters():
            p.requires_grad_(True)

    def normalized_rpy_loss(self, rpy_pred: torch.Tensor, rpy_true: torch.Tensor) -> torch.Tensor:
        """
        Compute loss between roll-pitch-yaw angles with normalized differences to [-π, π].
        This handles the periodicity of angles and ensures the shortest angular distance.
        
        Args:
            rpy_pred: Predicted RPY angles (B, 3) [roll, pitch, yaw]
            rpy_true: True RPY angles (B, 3) [roll, pitch, yaw]
            
        Returns:
            Mean squared error of normalized angular differences
        """
        # Compute the difference between predicted and true angles
        diff = rpy_pred - rpy_true
        
        # Normalize differences to [-π, π] range
        # This handles the periodicity of angles (e.g., 359° and 1° should have diff = 2° not 358°)
        normalized_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
        
        # Compute MSE loss on the normalized differences
        return F.mse_loss(normalized_diff, torch.zeros_like(normalized_diff))

    def supervised_step(
        self,
        batch_x: torch.Tensor,          # (B,14) [v, w, q, u]
        batch_y: torch.Tensor,          # (B,10) [v', w', q']
        optimizer: torch.optim.Optimizer,
        w_v: float = 1.0, w_w: float = 1.0, w_q: float = 1.0
    ) -> Dict[str, float]:
        """
        One offline training step (supervised on next-step targets).
        Combines MSE on velocities and a geodesic loss on quaternion.
        """
        is_train = optimizer is not None
        # print("is_train", is_train)
        self.train(is_train)
        pred = self(batch_x)
        # v_p, w_p, q_p_raw = self.split_outputs_rpy(pred)
        # v_t, w_t, q_t     = self.split_outputs_rpy(batch_y)


        v_p, w_p, rpy_p = self.split_outputs_rpy(pred)
        v_t, w_t, rpy_t = self.split_outputs_rpy(batch_y)
        # MSE for v, ω
        loss_v = F.mse_loss(v_p, v_t)
        loss_w = F.mse_loss(w_p, w_t)
        loss_rpy = F.mse_loss(rpy_p, rpy_t)
        loss = w_v*loss_v + w_w*loss_w + w_q*loss_rpy

        if not is_train:
            # validation path: no backward/step
            return {"loss": float(loss.item()), 
            "lv": float(loss_v.item()), 
            "lw": float(loss_w.item()), 
            "lq": float(loss_rpy.item())}

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return {"loss": float(loss.item()), "lv": float(loss_v.item()), "lw": float(loss_w.item()), "lq": float(loss_rpy.item())}

    def sample(self, actions,
                model_state,
                deterministic=None,
                rng=None): #next_observs,pred_rewards,pred_terminals,next_model_state,

        # obs = torch.from_numpy(model_state["obs"]).float().to(self.device)   
        model_input = torch.cat([model_state["obs"], actions], dim=-1)
        model_input_norm = (model_input - self.x_mean) / self.x_std
        # print("model_input_pitch", model_state["obs"][7].item(),"norm_pitch", model_input_norm[7].item())
        with torch.no_grad():
            next_observs_norm = self.forward(model_input_norm)
            pred_rewards = None
            pred_terminals = None
            next_model_state = model_state
        # next_observs[:, 6:10] = self.nonunit_quat_to_unit_quat_efficient(next_observs[:, 6:10])
        # next_observs[:, 6:10] = self.nonunit_quat_to_unit_quat(next_observs[:, 6:10])
        next_observs = next_observs_norm * self.y_std + self.y_mean
        # print("next_observs_norm_pitch", next_observs_norm[7].item(),"next_observs_pitch", next_observs[7].item())
        new_model_state = {}
        new_model_state["obs"] = next_observs
        
    
        return next_observs, None, None, new_model_state
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))
    
    def quat_to_R_nonunit(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Map (possibly non-unit) quaternion(s) to rotation matrix/matrices via:
            R = Q / ||q||^2
        q: (B,4) with components [qw, qx, qy, qz]
        returns R: (B,3,3)
        """
        qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        qw2, qx2, qy2, qz2 = qw*qw, qx*qx, qy*qy, qz*qz

        r00 = qw2 + qx2 - qy2 - qz2
        r01 = 2*(qx*qy - qw*qz)
        r02 = 2*(qx*qz + qw*qy)

        r10 = 2*(qx*qy + qw*qz)
        r11 = qw2 + qy2 - qx2 - qz2
        r12 = 2*(qy*qz - qw*qx)

        r20 = 2*(qx*qz - qw*qy)
        r21 = 2*(qy*qz + qw*qx)
        r22 = qw2 + qz2 - qx2 - qy2

        Qmat = torch.stack([
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ], dim=-2)  # (B,3,3)

        norm2 = (qw2 + qx2 + qy2 + qz2).clamp_min(eps).unsqueeze(-1).unsqueeze(-1)
        return Qmat / norm2

    def R_to_unit_quat(self, R: torch.Tensor, eps: float = 1e-8, make_qw_positive: bool = True) -> torch.Tensor:
        """
        Convert batch of rotation matrices to unit quaternions [qw, qx, qy, qz].
        R: (B,3,3)
        out: (B,4)
        """
        B = R.shape[0]
        trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
        q = torch.empty(B, 4, dtype=R.dtype, device=R.device)

        tr_pos = trace > 0
        if tr_pos.any():
            t = trace[tr_pos]
            s = torch.sqrt(t + 1.0 + eps) * 2
            qw = 0.25 * s
            qx = (R[tr_pos, 2, 1] - R[tr_pos, 1, 2]) / s
            qy = (R[tr_pos, 0, 2] - R[tr_pos, 2, 0]) / s
            qz = (R[tr_pos, 1, 0] - R[tr_pos, 0, 1]) / s
            q[tr_pos] = torch.stack([qw, qx, qy, qz], dim=-1)

        not_tr_pos = ~tr_pos
        if not_tr_pos.any():
            Rnt = R[not_tr_pos]
            diag = torch.stack([Rnt[:, 0, 0], Rnt[:, 1, 1], Rnt[:, 2, 2]], dim=-1)
            idx = torch.argmax(diag, dim=-1)

            qw = torch.empty_like(idx, dtype=R.dtype).to(R.device)
            qx = torch.empty_like(qw)
            qy = torch.empty_like(qw)
            qz = torch.empty_like(qw)

            mask0 = idx == 0
            if mask0.any():
                R0 = Rnt[mask0]
                s = torch.sqrt(1.0 + R0[:, 0, 0] - R0[:, 1, 1] - R0[:, 2, 2] + eps) * 2
                qx[mask0] = 0.25 * s
                qw[mask0] = (R0[:, 2, 1] - R0[:, 1, 2]) / s
                qy[mask0] = (R0[:, 0, 1] + R0[:, 1, 0]) / s
                qz[mask0] = (R0[:, 0, 2] + R0[:, 2, 0]) / s

            mask1 = idx == 1
            if mask1.any():
                R1 = Rnt[mask1]
                s = torch.sqrt(1.0 - R1[:, 0, 0] + R1[:, 1, 1] - R1[:, 2, 2] + eps) * 2
                qy[mask1] = 0.25 * s
                qw[mask1] = (R1[:, 0, 2] - R1[:, 2, 0]) / s
                qx[mask1] = (R1[:, 0, 1] + R1[:, 1, 0]) / s
                qz[mask1] = (R1[:, 1, 2] + R1[:, 2, 1]) / s

            mask2 = idx == 2
            if mask2.any():
                R2 = Rnt[mask2]
                s = torch.sqrt(1.0 - R2[:, 0, 0] - R2[:, 1, 1] + R2[:, 2, 2] + eps) * 2
                qz[mask2] = 0.25 * s
                qw[mask2] = (R2[:, 1, 0] - R2[:, 0, 1]) / s
                qx[mask2] = (R2[:, 0, 2] + R2[:, 2, 0]) / s
                qy[mask2] = (R2[:, 1, 2] + R2[:, 2, 1]) / s

            q[not_tr_pos] = torch.stack([qw, qx, qy, qz], dim=-1)

        q = q / (q.norm(dim=-1, keepdim=True).clamp_min(eps))
        if make_qw_positive:
            sign = torch.where(q[:, 0:1] < 0, -1.0, 1.0)
            q = q * sign
        return q

    def nonunit_quat_to_unit_quat(self, q: torch.Tensor) -> torch.Tensor:
        """
        Convert a possibly non-unit quaternion to a proper unit quaternion
        by going through rotation matrix representation.
        q: (B,4) quaternions [qw,qx,qy,qz]
        returns: (B,4) unit quaternions
        """
        R = self.quat_to_R_nonunit(q)        # fix non-unit quaternion -> rotation matrix
        q_unit = self.R_to_unit_quat(R)      # rotation matrix -> unit quaternion
        return q_unit
            
    def nonunit_quat_to_unit_quat_efficient(self, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """
        Convert a possibly non-unit quaternion to a proper unit quaternion
        by normalizing it.
        q: (B,4) quaternions [qw,qx,qy,qz]
        returns: (B,4) unit quaternions
        """
        # Normalize the quaternion to have unit length.
        # F.normalize is a robust way to do this, handling near-zero norms.
        q_unit = F.normalize(q, p=2, dim=-1, eps=eps)

        # Optional: ensure the scalar part (qw) is positive for a canonical representation,
        # since q and -q represent the same rotation.
        sign = torch.where(q_unit[:, 0:1] < 0, -1.0, 1.0)
        return q_unit * sign
    
    def attach_stats(self, x_mean, x_std, y_mean, y_std):
        # copy into the already-registered buffers; preserves registration/device
        self.x_mean.resize_(x_mean.shape).copy_(x_mean)
        self.x_std.resize_(x_std.shape).copy_(x_std.clamp_min(1e-8))
        self.y_mean.resize_(y_mean.shape).copy_(y_mean)
        self.y_std.resize_(y_std.shape).copy_(y_std.clamp_min(1e-8))

 