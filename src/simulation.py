import mujoco as mj
from mujoco.glfw import glfw
import numpy as np

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900
WINDOW_TITLE = "Mochi Simulation"
PIP_WIDTH = 320
PIP_HEIGHT = 240
PIP_MARGIN = 20


class Simulation:

    def __init__(self, model, data, controller):
        self.model = model
        self.data = data
        self.controller = controller
        self.cam = mj.MjvCamera()
        self.opt = mj.MjvOption()

        # --- GLFW and MuJoCo Visualization Init ---
        glfw.init()
        self.window = glfw.create_window(
            WINDOW_WIDTH, WINDOW_HEIGHT, WINDOW_TITLE, None, None
        )
        glfw.make_context_current(self.window)
        glfw.swap_interval(1)

        mj.mjv_defaultCamera(self.cam)
        mj.mjv_defaultOption(self.opt)
        self.scene_main = mj.MjvScene(self.model, maxgeom=10000)
        self.scene_pip = mj.MjvScene(self.model, maxgeom=10000)
        self.context = mj.MjrContext(self.model, mj.mjtFontScale.mjFONTSCALE_150.value)

        # --- PiP Camera Setup ---
        self.pip_cam = mj.MjvCamera()
        self.pip_cam.type = mj.mjtCamera.mjCAMERA_FIXED
        self.pip_cam.fixedcamid = self.model.camera("nicla_vision").id

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

        # 3. Render Sensor Overlay
        # Get formatted sensor data
        sensor_data_formatted = self.controller.get_sensor_readings()
        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            main_viewport,
            sensor_data_formatted,
            None,
            self.context,
        )

    def run(self):
        """Starts the main simulation loop."""
        while not glfw.window_should_close(self.window):
            time_prev = self.data.time
            while self.data.time - time_prev < 1.0 / 60.0:
                mj.mj_step(self.model, self.data)

            # Render the frame
            self._render_frame()

            # Swap buffers and poll events
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        glfw.terminate()
