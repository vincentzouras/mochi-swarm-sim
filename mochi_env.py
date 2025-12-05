import gymnasium as gym
import numpy as np
import mujoco as mj
import os
from gymnasium import spaces
from scipy.spatial.transform import Rotation as R

# Define mappings based on your XML names
ACTUATOR_NAMES = ["motor_left_thrust", "motor_right_thrust", "motors_servo"]
SENSOR_NAMES = {
    "accel": "imu_lin_vel",  # Using vel as proxy for state
    "gyro": "imu_ang_vel",
    "quat": "imu_quat",
    "pos": "imu_pos",
}


class MochiHoverEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    Goal: Hover at z=2.0 meters.
    Constraints: 1D Control (Vertical Effort).
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(MochiHoverEnv, self).__init__()

        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "models/mochi.xml")
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        # --- ACTION SPACE: 1 DIMENSION ---
        # Value [-1, 1] represents "Vertical Effort"
        # Positive = Fly Up (Servo +90)
        # Negative = Fly Down (Servo -90)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Observation Space: [z, vz]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
        )

        self.target_height = 2.0
        self.render_mode = render_mode
        self.dt = 0.01  # Matches your XML timestep

        # Cache IDs for faster access
        self.actuator_ids = [
            mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_ACTUATOR, n)
            for n in ACTUATOR_NAMES
        ]

        # Cache sensor ids and address ranges for efficient reads
        self.sensor_ids = {}
        self.sensor_ranges = {}
        for key, name in SENSOR_NAMES.items():
            try:
                sid = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_SENSOR, name)
                adr = self.model.sensor_adr[sid]
                dim = self.model.sensor_dim[sid]
                self.sensor_ids[key] = sid
                self.sensor_ranges[key] = (adr, dim)
            except Exception:
                self.sensor_ids[key] = None
                self.sensor_ranges[key] = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)

        # Randomize start altitude slightly
        self.data.qpos[2] = 1.0 + np.random.uniform(-0.5, 0.5)

        # Apply changes
        mj.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # 1. Parse 1D Action
        raw_val = action[0]

        # 2. Determine Servo & Thrust based on sign
        if raw_val >= 0:
            # Command > 0: Thrust UP against gravity
            servo_angle = np.pi / 2
            thrust_mag = raw_val  # 0 to 1
        else:
            # Command < 0: Thrust DOWN (power dive)
            servo_angle = -np.pi / 2
            thrust_mag = -raw_val  # Flip sign to make magnitude positive

        # 3. Apply Constraints (Equal Thrust)
        left_thrust = thrust_mag
        right_thrust = thrust_mag

        # 4. Apply to MuJoCo
        self.data.ctrl[self.actuator_ids[0]] = left_thrust
        self.data.ctrl[self.actuator_ids[1]] = right_thrust
        self.data.ctrl[self.actuator_ids[2]] = servo_angle

        # 5. Step Physics
        mj.mj_step(self.model, self.data)

        # 6. Get Observation
        obs = self._get_obs()
        z, vz = obs

        # 7. Calculate Reward
        dist_to_target = abs(z - self.target_height)

        # Reward A: Height (Exponential for precision)
        r_height = np.exp(-2.0 * dist_to_target)

        # Reward B: Survival (Keep flying!)
        r_survival = 0.1

        # Reward C: Stability (Penalty for horizontal drift)
        lin_vel = self.data.sensor("imu_lin_vel").data
        vx, vy = lin_vel[0], lin_vel[1]
        r_drift = -0.1 * (abs(vx) + abs(vy))

        reward = r_height + r_survival + r_drift

        # 8. Check Termination
        terminated = False
        if z > 5.0 or z < 0.1:  # Hit ceiling or floor
            terminated = True
            reward = -10.0  # Hard crash penalty

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        def read_sensor(name_key):
            rng = self.sensor_ranges.get(name_key)
            if not rng:
                raise RuntimeError(f"Required sensor '{name_key}' not found")
            adr, dim = rng
            return self.data.sensordata[adr : adr + dim]

        pos = read_sensor("pos")
        lin_vel = read_sensor("accel")  # This is imu_lin_vel

        z = float(pos[2])
        vz = float(lin_vel[2])

        # We ignore everything else now!
        return np.array([z, vz], dtype=np.float32)

    def render(self):
        pass
