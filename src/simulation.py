import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

from .definitions import AXLE, THRUST_LEFT, THRUST_RIGHT, Action, State


PIP_WIDTH = 240
PIP_HEIGHT = 160
PIP_MARGIN = 20
CAMERA = "nicla_vision"
ASSEMBLY = "assembly"


class Simulation:

    def __init__(self, model, data, controller):
        self.model = model
        self.data = data
        self.controller = controller
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()
        self.camera_follow = True

        # Trajectory recording (world XYZ of main body)
        self._assembly_id = self.model.body(ASSEMBLY).id
        self.traj_x = []
        self.traj_y = []
        self.traj_z = []

        # --- GLFW and MuJoCo Visualization Init ---
        glfw.init()
        glfw.window_hint(glfw.MAXIMIZED, glfw.TRUE)
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.window = glfw.create_window(
            mode.size.width, mode.size.height, "Mochi Simulation", None, None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.cam.distance = 20.0
        self.cam.azimuth = 60
        self.cam.elevation = -20

        self.scene_main = mj.MjvScene(self.model, maxgeom=10000)
        self.scene_pip = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # --- PiP Camera Setup ---
        self.pip_cam = mj.MjvCamera()
        self.pip_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.pip_cam.fixedcamid = self.model.camera(CAMERA).id

        # --- Input State Variables ---
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.lastx = 0
        self.lasty = 0

        # --- Set Callbacks ---
        # We pass instance methods to GLFW
        glfw.set_key_callback(self.window, self._keyboard_callback)
        glfw.set_cursor_pos_callback(self.window, self._mouse_move_callback)
        glfw.set_mouse_button_callback(self.window, self._mouse_button_callback)
        glfw.set_scroll_callback(self.window, self._scroll_callback)

        # Set the MuJoCo control callback to the controller's method
        mj.set_mjcb_control(self.controller.control_step)

    # --- Internal Callback Functions ---
    # These are prefixed with _ to show they are "private"

    def _keyboard_callback(self, window, key, scancode, act, mods):
        """Handles keyboard input."""
        # Reset simulation
        if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
            self.camera_follow = True
            mj.mj_resetData(self.model, self.data)
            mj.mj_forward(self.model, self.data)
            return

        # Pass all other key events to the controller
        self.controller.update_key_state(key, act)

    def _mouse_button_callback(self, window, button, act, mods):
        """Handles mouse button input for camera control."""
        self.button_left = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
        )
        self.button_middle = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS
        )
        self.button_right = (
            glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
        )
        glfw.get_cursor_pos(window)

    def _mouse_move_callback(self, window, xpos, ypos):
        """Handles mouse movement for camera control."""
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx, self.lasty = xpos, ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        width, height = glfw.get_window_size(window)
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        if self.button_right:
            self.camera_follow = False
            action = (
                mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
            )
        elif self.button_left:
            action = (
                mj.mjtMouse.mjMOUSE_ROTATE_H
                if mod_shift
                else mj.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mj.mjtMouse.mjMOUSE_ZOOM

        mj.mjv_moveCamera(
            self.model, action, dx / height, dy / height, self.scene_main, self.cam
        )

    def _scroll_callback(self, window, xoffset, yoffset):
        """Handles mouse scroll for camera zoom."""
        mj.mjv_moveCamera(
            self.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene_main,
            self.cam,
        )

    def _render_frame(self):
        """Renders one frame of the simulation, including PiP and overlays."""

        if self.camera_follow:
            pos = self.data.xpos[self._assembly_id]
            self.cam.lookat[:] = pos

        viewport_width, viewport_height = glfw.get_framebuffer_size(self.window)
        main_viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

        # 1. Render Main Scene
        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene_main,
        )
        mj.mjr_render(main_viewport, self.scene_main, self.context)

        # 2. Render PiP Scene
        pip_x = viewport_width - PIP_WIDTH - PIP_MARGIN
        pip_y = PIP_MARGIN
        pip_viewport = mj.MjrRect(pip_x, pip_y, PIP_WIDTH, PIP_HEIGHT)

        mj.mjv_updateScene(
            self.model,
            self.data,
            self.opt,
            None,
            self.pip_cam,
            mj.mjtCatBit.mjCAT_ALL.value,
            self.scene_pip,
        )
        mj.mjr_render(pip_viewport, self.scene_pip, self.context)

        # 3. Render Sensor Display
        sensor_data = self.controller.senses
        labels = [name.lower() for name in State.__members__ if name != "NUM_STATES"]
        sensors_formatted = "\n".join([f"{val:8.3f}" for val in sensor_data])
        sensors_labels = "\n".join(labels)
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            main_viewport,
            sensors_formatted,
            sensors_labels,
            self.context,
        )

        # 4. Render additional information
        armed = "ARMED" if self.controller.action_states[Action.ARMED] else "DISARMED"
        info_formatted = (
            f"{armed}\n"
            f"{self.controller.state_machine.current_state.target_height:8.3f}\n"
            f"{sensor_data[State.Z_ALTITUDE]:8.3f}\n"
            f"{self.controller.state_machine.current_state.target_yaw:8.3f}\n"
            f"{sensor_data[State.Z_YAW]:8.3f}\n"
            f"{self.controller.state_machine.current_state.target_thrust:8.3f}\n"
            f"{self.data.actuator(THRUST_LEFT).ctrl[0]:8.3f}\n"
            f"{self.data.actuator(THRUST_RIGHT).ctrl[0]:8.3f}\n"
            f"{self.data.joint(AXLE).qpos[0]:8.3f}"
        )
        info_labels = (
            f"Status\n"
            f"Altitude Target\n"
            f"Altitude Actual\n"
            f"Yaw Target\n"
            f"Yaw Actual\n"
            f"Target Thrust\n"
            f"Motor L Thrust\n"
            f"Motor R Thrust\n"
            f"Servo Angle"
        )
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_BOTTOMLEFT,
            main_viewport,
            info_formatted,
            info_labels,
            self.context,
        )

    def run(self):
        """Starts the main simulation loop."""
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < 1.0 / 60.0:
                mj.mj_step(self.model, self.data)

            # Record trajectory point once per rendered frame
            pos = self.data.xpos[self._assembly_id]
            self.traj_x.append(float(pos[0]))
            self.traj_y.append(float(pos[1]))
            self.traj_z.append(float(pos[2]))

            # Render the frame
            self._render_frame()

            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        self.stop()

    def stop(self):
        glfw.terminate()
        try:
            if len(self.traj_x) > 1:
                self._plot_3d_trajectory()
                self._plot_xy_trajectory()
        except Exception as e:
            print(f"Trajectory plotting skipped ({e})")

    def _plot_3d_trajectory(self):
        """Render a 3D plot of the recorded trajectory."""
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            self.traj_x,
            self.traj_y,
            self.traj_z,
            "-",
            linewidth=1.5,
            label="trajectory",
        )
        ax.scatter(
            self.traj_x[0],
            self.traj_y[0],
            self.traj_z[0],
            c="green",
            s=40,
            label="start",
        )
        ax.scatter(
            self.traj_x[-1],
            self.traj_y[-1],
            self.traj_z[-1],
            c="red",
            s=40,
            label="end",
        )

        xs = np.array(self.traj_x)
        ys = np.array(self.traj_y)
        zs = np.array(self.traj_z)
        x_range = xs.max() - xs.min()
        y_range = ys.max() - ys.min()
        z_range = zs.max() - zs.min()
        max_range = max(x_range, y_range, z_range, 1e-9)
        ax.set_box_aspect(
            (x_range / max_range, y_range / max_range, z_range / max_range)
        )

        ax.set_title("Robot 3D trajectory")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def _plot_xy_trajectory(self):
        """Render a 2D top-down plot of the recorded trajectory."""
        import matplotlib.pyplot as plt

        fig = plt.figure(figsize=(6, 6))
        ax2 = fig.add_subplot(111)
        ax2.plot(self.traj_x, self.traj_y, "-", linewidth=1.5, label="trajectory")
        ax2.scatter(self.traj_x[0], self.traj_y[0], c="green", s=40, label="start")
        ax2.scatter(self.traj_x[-1], self.traj_y[-1], c="red", s=40, label="end")
        ax2.set_aspect("equal", adjustable="box")
        ax2.set_title("Robot XY trajectory")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="best")
        plt.tight_layout()
        plt.show()
