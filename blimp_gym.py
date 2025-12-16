"""
Gym-style environment wrapper for the MuJoCo blimp simulation.
Provides a clean interface for reinforcement learning and control algorithms.
"""

import numpy as np
import mujoco as mj
from scipy.spatial.transform import Rotation as R
from typing import Tuple, Optional, Dict, Any
import os

from src.definitions import (
    SERVO,
    THRUST_LEFT,
    THRUST_RIGHT,
    IMU_POS,
    IMU_LIN_VEL,
    IMU_ANG_VEL,
    IMU_QUAT,
    State,
)


class BlimpGymEnv:
    """
    Gym-style environment for the MuJoCo blimp simulation.
    
    State space: [x_vel, y_vel, z_vel, x_roll, y_pitch, z_yaw, x_roll_rate, y_pitch_rate, z_yaw_rate]
    Action space: [left_thrust, right_thrust, servo_angle]
        - left_thrust: [0, 1]
        - right_thrust: [0, 1]
        - servo_angle: [-π, π]
    """
    
    def __init__(
        self,
        model_xml_path: str = "models/mochi.xml",
        headless: bool = True,
        dt: float = 0.01,
        max_episode_steps: Optional[int] = None,
        render_mode: Optional[str] = None,
    ):
        """
        Initialize the blimp gym environment.
        
        Args:
            model_xml_path: Path to MuJoCo XML model file
            headless: If True, run without GUI (faster)
            dt: Time step for simulation (should match model timestep)
            max_episode_steps: Maximum steps per episode (None = unlimited)
            render_mode: Rendering mode ('human', 'rgb_array', or None)
        """
        self.headless = headless
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.render_mode = render_mode
        self._step_count = 0
        
        # Resolve model path - try multiple locations
        original_path = model_xml_path
        if not os.path.isabs(model_xml_path):
            # Try 1: Relative to current working directory
            cwd_path = os.path.join(os.getcwd(), model_xml_path)
            if os.path.exists(cwd_path):
                model_xml_path = os.path.abspath(cwd_path)
            # Try 2: Absolute path from relative
            elif os.path.exists(os.path.abspath(model_xml_path)):
                model_xml_path = os.path.abspath(model_xml_path)
            # Try 3: Relative to this file's directory
            else:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                parent_dir = os.path.dirname(script_dir)  # Go up one level from blimp_mppi/
                alt_path = os.path.join(parent_dir, model_xml_path)
                if os.path.exists(alt_path):
                    model_xml_path = os.path.abspath(alt_path)
        
        if not os.path.exists(model_xml_path):
            raise FileNotFoundError(
                f"Model file not found: {original_path}\n"
                f"Tried paths:\n"
                f"  - {os.path.join(os.getcwd(), original_path)}\n"
                f"  - {os.path.abspath(original_path)}\n"
                f"Current working directory: {os.getcwd()}\n"
                f"Please provide an absolute path or ensure the file exists relative to the working directory."
            )
        
        # Load MuJoCo model with better error handling
        # MuJoCo resolves relative paths in XML (like meshdir) relative to the XML file's directory
        # So we need to change to that directory temporarily
        model_dir = os.path.dirname(os.path.abspath(model_xml_path))
        original_cwd = os.getcwd()
        
        print(f"Loading MuJoCo model from: {model_xml_path}")
        print(f"Model directory: {model_dir}")
        print(f"Current working directory: {original_cwd}")
        
        try:
            # Change to model directory so relative paths in XML resolve correctly
            os.chdir(model_dir)
            xml_filename = os.path.basename(model_xml_path)
            self.model = mj.MjModel.from_xml_path(xml_filename)
            self.data = mj.MjData(self.model)
            # Change back to original directory
            os.chdir(original_cwd)
            print("Model loaded successfully")
        except ValueError as e:
            # Make sure we change back even on error
            os.chdir(original_cwd)
            # MuJoCo often wraps errors in ValueError with "engine error"
            import traceback
            error_msg = (
                f"Failed to load MuJoCo model from {model_xml_path}\n"
                f"MuJoCo error: {str(e)}\n"
                f"This might be due to:\n"
                f"  - Missing mesh files referenced in the XML\n"
                f"  - Invalid XML syntax\n"
                f"  - Path issues with relative paths in the XML\n"
                f"Full traceback:\n{traceback.format_exc()}"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            # Make sure we change back even on error
            os.chdir(original_cwd)
            import traceback
            error_msg = (
                f"Failed to load MuJoCo model from {model_xml_path}\n"
                f"Error: {str(e)}\n"
                f"Type: {type(e).__name__}\n"
                f"Full traceback:\n{traceback.format_exc()}"
            )
            print(error_msg)
            raise RuntimeError(error_msg) from e
        
        # Initialize rendering if needed
        self._render_setup = False
        # Setup rendering if not headless (regardless of render_mode)
        if not headless:
            self._setup_rendering()
        
        # State and action spaces
        self.observation_space = self._get_observation_space()
        self.action_space = self._get_action_space()
        
        # Internal state tracking
        self._senses = np.zeros(State.NUM_STATES)
        
    def _get_observation_space(self) -> Dict[str, Any]:
        """Get observation space definition."""
        return {
            'shape': (9,),
            'dtype': np.float32,
            'low': np.array([-np.inf] * 9, dtype=np.float32),
            'high': np.array([np.inf] * 9, dtype=np.float32),
        }
    
    def _get_action_space(self) -> Dict[str, Any]:
        """Get action space definition."""
        return {
            'shape': (3,),
            'dtype': np.float32,
            'low': np.array([0.0, 0.0, -np.pi], dtype=np.float32),
            'high': np.array([1.0, 1.0, np.pi], dtype=np.float32),
        }
    
    def _setup_rendering(self):
        """Setup rendering components if not headless."""
        if self._render_setup:
            return
            
        from mujoco.glfw import glfw
        from src.simulation import Simulation
        
        # We'll use a minimal rendering setup
        # For full rendering, you can use the Simulation class
        self._glfw = glfw
        self._render_setup = True
        
        if not self.headless:
            glfw.init()
            glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
            monitor = glfw.get_primary_monitor()
            mode = glfw.get_video_mode(monitor)
            self.window = glfw.create_window(
                mode.size.width, mode.size.height, "Blimp Gym", None, None
            )
            if not self.window:
                raise RuntimeError("Failed to create GLFW window")
            glfw.make_context_current(self.window)
            glfw.swap_interval(1)
            
            self.cam = mj.MjvCamera()
            self.opt = mj.MjvOption()
            self.scene = mj.MjvScene(self.model, maxgeom=10000)
            self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)
            
            mj.mjv_defaultCamera(self.cam)
            mj.mjv_defaultOption(self.opt)
            self.cam.distance = 20.0
            self.cam.azimuth = 60
            self.cam.elevation = -20
            
            # Do an initial render to show the window
            self.render()
            print("Rendering window created and initialized")
    
    def _sense(self):
        """
        Update internal sensor readings.
        Same as controller.py _sense() method.
        """
        # Z altitude
        self._senses[State.Z_ALTITUDE] = self.data.sensor(IMU_POS).data.copy()[2]
        # Z altitude velocity
        self._senses[State.Z_ALTITUDE_VEL] = self.data.sensor(IMU_LIN_VEL).data.copy()[2]
        
        # Convert quaternion to Euler angles
        quat = self.data.sensor(IMU_QUAT).data.copy()  # [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)  # in radians
        
        self._senses[State.X_ROLL] = roll
        self._senses[State.Y_PITCH] = pitch
        self._senses[State.Z_YAW] = yaw
        
        # Angular velocities
        ang_vel = self.data.sensor(IMU_ANG_VEL).data.copy()
        self._senses[State.X_ROLL_RATE] = ang_vel[0]
        self._senses[State.Y_PITCH_RATE] = ang_vel[1]
        self._senses[State.Z_YAW_RATE] = ang_vel[2]
    
    def _get_global_velocity(self) -> Tuple[float, float]:
        """
        Get global velocity components (x, y) in world frame.
        Same as controller.py get_global_velocity() method.
        """
        lin_vel = self.data.sensor(IMU_LIN_VEL).data.copy()  # [vx, vy, vz]
        x_vel = lin_vel[0]
        y_vel = lin_vel[1]
        return x_vel, y_vel
    
    def _extract_state(self) -> np.ndarray:
        """
        Extract state vector from simulator.
        State format: [x_vel, y_vel, z_vel, x_roll, y_pitch, z_yaw, x_roll_rate, y_pitch_rate, z_yaw_rate]
        Same format as controller.py capture_next_state().
        """
        self._sense()
        x_vel, y_vel = self._get_global_velocity()
        z_vel = self._senses[State.Z_ALTITUDE_VEL]
        
        # Reorder: velocities first, then orientation, then angular velocities
        state = np.array([
            x_vel,                                    # x_vel
            y_vel,                                    # y_vel
            z_vel,                                    # z_vel (z_altitude_vel)
            self._senses[State.X_ROLL],                # x_roll
            self._senses[State.Y_PITCH],               # y_pitch
            self._senses[State.Z_YAW],                 # z_yaw
            self._senses[State.X_ROLL_RATE],           # x_roll_rate
            self._senses[State.Y_PITCH_RATE],          # y_pitch_rate
            self._senses[State.Z_YAW_RATE]             # z_yaw_rate
        ], dtype=np.float32)
        
        return state
    
    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset the environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            options: Additional options (e.g., initial position)
            
        Returns:
            observation: Initial state observation
            info: Additional information
        """
        if seed is not None:
            np.random.seed(seed)
        
        # Reset MuJoCo simulation
        mj.mj_resetData(self.model, self.data)
        mj.mj_forward(self.model, self.data)
        
        # Set initial position if provided
        if options is not None and 'initial_position' in options:
            initial_pos = options['initial_position']
            if len(initial_pos) == 3:
                # Set position of the blimp body (assembly)
                assembly_id = self.model.body("assembly").id
                self.data.xpos[assembly_id] = initial_pos
        
        # Reset step counter
        self._step_count = 0
        
        # Extract initial state
        observation = self._extract_state()
        info = {
            'step_count': self._step_count,
            'position': self.data.sensor(IMU_POS).data.copy()  # [x, y, z] cartesian coordinates
        }
        
        return observation, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Step the environment forward by one timestep.
        
        Args:
            action: Action array [left_thrust, right_thrust, servo_angle]
                - left_thrust: [0, 1]
                - right_thrust: [0, 1]
                - servo_angle: [-π, π]
        
        Returns:
            observation: Next state observation
            reward: Reward (default 0, can be overridden)
            terminated: Whether episode terminated (goal reached, etc.)
            truncated: Whether episode truncated (max steps, etc.)
            info: Additional information
        """
        # Clip action to valid ranges
        action = np.clip(action, self.action_space['low'], self.action_space['high'])
        
        # Apply action to actuators
        self.data.actuator(THRUST_LEFT).ctrl = action[0]
        self.data.actuator(THRUST_RIGHT).ctrl = action[1]
        self.data.actuator(SERVO).ctrl = action[2]
        
        # Step simulation forward
        mj.mj_step(self.model, self.data)
        
        # Extract next state
        observation = self._extract_state()
        
        # Update step counter
        self._step_count += 1
        
        # Check termination conditions
        terminated = False  # Can be customized based on task
        truncated = False
        if self.max_episode_steps is not None:
            truncated = self._step_count >= self.max_episode_steps
        
        # Default reward (can be customized)
        reward = 0.0
        
        # Additional info
        info = {
            'step_count': self._step_count,
            'position': self.data.sensor(IMU_POS).data.copy(),
            'velocity': self.data.sensor(IMU_LIN_VEL).data.copy(),
        }
        
        # Render if not headless and rendering is set up
        if not self.headless and self._render_setup:
            self.render()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """
        Render the current state of the environment.
        """
        if self.headless or not self._render_setup:
            return
        
        from mujoco.glfw import glfw
        
        # Check if window should close
        if glfw.window_should_close(self.window):
            return
        
        # Update camera to follow the blimp (optional - can be removed if you want fixed camera)
        try:
            assembly_id = self.model.body("assembly").id
            pos = self.data.xpos[assembly_id]
            self.cam.lookat[:] = pos
        except:
            pass  # If assembly body not found, use default camera position
        
        # Update scene
        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)
        
        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene,
        )
        mj.mjr_render(viewport, self.scene, self.context)
        
        # Swap buffers and poll events
        glfw.swap_buffers(self.window)
        glfw.poll_events()
    
    def close(self):
        """
        Clean up resources.
        """
        if self._render_setup and not self.headless:
            from mujoco.glfw import glfw
            glfw.terminate()
            self._render_setup = False
    
    def get_position(self) -> np.ndarray:
        """
        Get current position [x, y, z] of the blimp.
        
        Returns:
            Position array [x, y, z]
        """
        return self.data.sensor(IMU_POS).data.copy()
    
    def get_velocity(self) -> np.ndarray:
        """
        Get current linear velocity [vx, vy, vz] of the blimp.
        
        Returns:
            Velocity array [vx, vy, vz]
        """
        return self.data.sensor(IMU_LIN_VEL).data.copy()
    
    def get_orientation(self) -> Tuple[float, float, float]:
        """
        Get current orientation (roll, pitch, yaw) in radians.
        
        Returns:
            Tuple of (roll, pitch, yaw) in radians
        """
        quat = self.data.sensor(IMU_QUAT).data.copy()  # [w, x, y, z]
        r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])  # scipy uses [x, y, z, w]
        roll, pitch, yaw = r.as_euler("xyz", degrees=False)
        return roll, pitch, yaw
    
    def __del__(self):
        """Cleanup on deletion."""
        self.close()

if __name__ == "__main__":
    # Create environment
    env = BlimpGymEnv(
        model_xml_path="models/mochi.xml",
        headless=False,  # Set False to see GUI
        dt=0.01,
        max_episode_steps=1000
    )

    # Reset environment
    obs, info = env.reset()

    # Step through environment
    import time
    for i in range(10000):
        action = np.array([0.25, 0.25, 3.14/2])  # [left_thrust, right_thrust, servo_angle]
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Small delay to allow rendering (only if not headless)
        if not env.headless:
            time.sleep(0.01)  # ~100 FPS
        
        if i % 100 == 0:
            print(f"Step {i}: Position = {info['position']}")
        
        if terminated or truncated:
            print("Episode over")
            obs, info = env.reset()

    # Cleanup
    env.close()