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
    Goal: Hover at z=2.0 meters with yaw=0.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super(MochiHoverEnv, self).__init__()

        # Load model
        model_path = os.path.join(os.path.dirname(__file__), "models/mochi.xml")
        # MuJoCo Python API: create model/data via class constructors
        self.model = mj.MjModel.from_xml_path(model_path)
        self.data = mj.MjData(self.model)

        # Action Space: [Left Thrust (0-1), Right Thrust (0-1), Servo Angle (-pi to pi)]
        # We normalize inputs to [-1, 1] for PPO stability, then scale them inside step()
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)

        # Observation Space: [z, vz, roll, pitch, yaw, wx, wy, wz]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
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
                # If a sensor is missing, record None to catch during _get_obs
                self.sensor_ids[key] = None
                self.sensor_ranges[key] = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mj.mj_resetData(self.model, self.data)

        # Randomize start slightly to make policy robust
        self.data.qpos[2] = 1.0 + np.random.uniform(
            -0.5, 0.5
        )  # Start z between 0.5 and 1.5
        mj.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(self, action):
        # 1. Scale actions from RL [-1, 1] to physical limits
        # Thrust: [-1, 1] -> [0, 1]
        left_thrust = (action[0] + 1) / 2
        right_thrust = (action[1] + 1) / 2
        # Servo: [-1, 1] -> [-pi, pi]
        servo_angle = action[2] * np.pi

        # 2. Apply to MuJoCo
        self.data.ctrl[self.actuator_ids[0]] = left_thrust
        self.data.ctrl[self.actuator_ids[1]] = right_thrust
        self.data.ctrl[self.actuator_ids[2]] = servo_angle

        # 3. Step Physics (Apply wind noise here if you added it earlier)
        mj.mj_step(self.model, self.data)

        # 4. Get Observation
        obs = self._get_obs()
        z, vz, roll, pitch, yaw, wx, wy, wz = obs

        # 5. Calculate Reward (THE FIX)

        # A. Height Reward: Exponential is better than linear.
        # It gives +1.0 at perfect height, +0.36 at 1m error, and +0.0 at huge error.
        dist_to_target = abs(z - self.target_height)
        r_height = np.exp(-2.0 * dist_to_target)

        # B. Survival Bonus: The "Stay Alive" incentive.
        # As long as it doesn't crash, it collects this small reward every single tick.
        r_survival = 0.1

        # C. Stability penalties (keep these small so they don't overpower survival)
        r_yaw = -0.1 * abs(yaw)
        r_stability = -0.1 * (abs(roll) + abs(pitch))

        # D. Energy Penalty: REMOVED (Commented out)
        # We want it to learn to fly first. We can optimize battery later.
        # r_energy = -(left_thrust**2 + right_thrust**2) * 0.1

        reward = r_height + r_survival + r_yaw + r_stability

        # 6. Check Termination
        terminated = False
        if z > 5.0 or z < 0.1:  # Hit ceiling or floor
            terminated = True
            # Crash penalty.
            # We replace the computed positive reward with a flat punishment.
            reward = -10.0

        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # Extract sensor data using cached address ranges into data.sensordata
        def read_sensor(name_key):
            rng = self.sensor_ranges.get(name_key)
            if not rng:
                raise RuntimeError(f"Required sensor '{name_key}' not found in model")
            adr, dim = rng
            return self.data.sensordata[adr : adr + dim]

        pos = read_sensor("pos")  # expected dim=3
        lin_vel = read_sensor("accel")  # using linear velocity as proxy for accel/state
        quat = read_sensor("quat")  # [w, x, y, z] per MuJoCo IMU convention
        ang_vel = read_sensor("gyro")  # [wx, wy, wz]

        z = float(pos[2])
        vz = float(lin_vel[2])

        # Convert quaternion to roll/pitch/yaw
        # MuJoCo IMU quaternion ordering is [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)

        return np.array(
            [z, vz, roll, pitch, yaw, ang_vel[0], ang_vel[1], ang_vel[2]],
            dtype=np.float32,
        )

    def render(self):
        # We can implement a simple viewer here if needed,
        # but usually we rely on the training script's callbacks or replay.
        pass
