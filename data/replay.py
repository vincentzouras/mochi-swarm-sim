import csv
import time
import math
import numpy as np
import mujoco as mj
from mujoco.glfw import glfw

# --- Configuration ---
MODEL_XML_PATH = "models/floor.xml"
CSV_FILE_PATH = "data/output.csv"

# The name of the root body that has the <freejoint> (moves the whole robot)
BODY_MAIN = "blimp"
# The name of the servo joint (rotates the motors)
JOINT_SERVO = "motors_axle"


# --- Interaction Control ---
class InteractionController:
    def __init__(self, model, cam, scene):
        self.model = model
        self.cam = cam
        self.scene = scene

        # Camera State
        self.lastx = 0
        self.lasty = 0
        self.button_left = False
        self.button_middle = False
        self.button_right = False

        # Playback State
        self.tracking = False
        self.paused = False
        self.playback_speed = 1.0
        self.sim_time = 0.0  # Current time in the replay
        self.min_time = 0.0  # Start time of log
        self.max_time = 0.0  # End time of log

    def mouse_button(self, window, button, act, mods):
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

    def mouse_move(self, window, xpos, ypos):
        dx = xpos - self.lastx
        dy = ypos - self.lasty
        self.lastx = xpos
        self.lasty = ypos

        if not (self.button_left or self.button_middle or self.button_right):
            return

        width, height = glfw.get_window_size(window)
        mod_shift = (
            glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS
            or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS
        )

        action = None
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

        if action is not None:
            mj.mjv_moveCamera(
                self.model, action, dx / height, dy / height, self.scene, self.cam
            )

    def scroll(self, window, xoffset, yoffset):
        mj.mjv_moveCamera(
            self.model,
            mj.mjtMouse.mjMOUSE_ZOOM,
            0.0,
            -0.05 * yoffset,
            self.scene,
            self.cam,
        )

    def key_callback(self, window, key, scancode, act, mods):
        if act != glfw.PRESS and act != glfw.REPEAT:
            return

        # Playback Controls
        if key == glfw.KEY_SPACE and act == glfw.PRESS:
            self.paused = not self.paused

        elif key == glfw.KEY_T and act == glfw.PRESS:
            self.tracking = not self.tracking

        elif key == glfw.KEY_R and act == glfw.PRESS:
            self.sim_time = self.min_time

        elif key == glfw.KEY_RIGHT:
            self.sim_time += 1.0  # Jump forward 1s

        elif key == glfw.KEY_LEFT:
            self.sim_time -= 1.0  # Jump back 1s

        elif key == glfw.KEY_UP:
            self.playback_speed += 0.25

        elif key == glfw.KEY_DOWN:
            self.playback_speed = max(0.25, self.playback_speed - 0.25)

        # Clamp time immediately so we don't seek out of bounds
        self.sim_time = max(self.min_time, min(self.sim_time, self.max_time))


# --- Math Helper ---
def euler_to_quaternion(roll, pitch, yaw):
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return [w, x, y, z]


def load_csv_data(filepath):
    data_points = []
    try:
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            for row in reader:
                try:
                    ts = float(row["timestamp"])
                    pos = [
                        float(row["pos_x"]),
                        float(row["pos_y"]),
                        float(row["pos_z"]),
                    ]

                    pitch = float(row["pitch"])
                    roll = float(row["roll"])
                    yaw = float(row["yaw"])
                    yaw -= math.pi / 2 * 3  # your existing rotation fix
                    quat = euler_to_quaternion(roll, pitch, yaw)

                    servo_deg = float(row.get("servo_angle", 0.0)) - 90.0
                    servo_rad = math.radians(servo_deg)

                    data_points.append(
                        {"time": ts, "pos": pos, "quat": quat, "servo": servo_rad}
                    )
                except ValueError:
                    continue
    except FileNotFoundError:
        print(f"Error: {filepath} not found.")
    return data_points


def main():
    # 1. Setup Model
    model = mj.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mj.MjData(model)

    # Find Joints
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, BODY_MAIN)
    jnt_adr = model.body_jntadr[body_id] if body_id != -1 else -1
    qpos_main_adr = model.jnt_qposadr[jnt_adr] if jnt_adr != -1 else 0

    servo_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, JOINT_SERVO)
    qpos_servo_adr = model.jnt_qposadr[servo_id] if servo_id != -1 else -1

    # 2. Setup Window
    glfw.init()
    window = glfw.create_window(
        1200, 900, "Mochi Replay (Space=Pause, T=Track, Arrows=Seek/Speed)", None, None
    )
    glfw.make_context_current(window)
    glfw.swap_interval(1)

    # 3. Setup Camera & Scene
    cam = mj.MjvCamera()
    mj.mjv_defaultCamera(cam)
    cam.distance = 5.0
    cam.lookat = [0, 0, 1]
    cam.azimuth = 45
    cam.elevation = -20

    opt = mj.MjvOption()
    mj.mjv_defaultOption(opt)
    scene = mj.MjvScene(model, maxgeom=10000)
    context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

    # 4. Attach Controls
    ctrl = InteractionController(model, cam, scene)
    glfw.set_cursor_pos_callback(window, ctrl.mouse_move)
    glfw.set_mouse_button_callback(window, ctrl.mouse_button)
    glfw.set_scroll_callback(window, ctrl.scroll)
    glfw.set_key_callback(window, ctrl.key_callback)

    # 5. Load Data
    flight_log = load_csv_data(CSV_FILE_PATH)
    if not flight_log:
        print("No valid data loaded.")
        return

    # Initialize playback state
    ctrl.min_time = flight_log[0]["time"]
    ctrl.max_time = flight_log[-1]["time"]
    ctrl.sim_time = ctrl.min_time

    last_render_time = time.time()
    idx = 0

    while not glfw.window_should_close(window):
        # Calculate Delta Time (wall clock)
        now = time.time()
        dt = now - last_render_time
        last_render_time = now

        # Advance Simulation Time (if not paused)
        if not ctrl.paused:
            ctrl.sim_time += dt * ctrl.playback_speed

        # Loop Check
        if ctrl.sim_time > ctrl.max_time:
            ctrl.sim_time = ctrl.min_time  # Loop to start
            idx = 0  # Hint to reset search

        # Clamp bounds (in case seek went out of bounds)
        ctrl.sim_time = max(ctrl.min_time, min(ctrl.sim_time, ctrl.max_time))

        # Find the correct index for the current sim_time
        # (Linear scan is fast enough for playback)
        while (
            idx < len(flight_log) - 1 and flight_log[idx + 1]["time"] <= ctrl.sim_time
        ):
            idx += 1
        while idx > 0 and flight_log[idx]["time"] > ctrl.sim_time:
            idx -= 1

        point = flight_log[idx]

        # --- UPDATE PHYSICS STATE ---
        data.qpos[qpos_main_adr : qpos_main_adr + 3] = point["pos"]
        data.qpos[qpos_main_adr + 3 : qpos_main_adr + 7] = point["quat"]
        if qpos_servo_adr != -1:
            data.qpos[qpos_servo_adr] = point["servo"]

        mj.mj_forward(model, data)

        # --- UPDATE CAMERA ---
        if ctrl.tracking:
            cam.lookat[:] = point["pos"]

        # --- RENDER ---
        viewport = mj.MjrRect(0, 0, *glfw.get_framebuffer_size(window))
        mj.mjv_updateScene(
            model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene
        )
        mj.mjr_render(viewport, scene, context)

        # Info Overlay
        servo_deg = math.degrees(point["servo"])
        status = "PAUSED" if ctrl.paused else f"PLAYING ({ctrl.playback_speed}x)"
        info = (
            f"{status}\n"
            f"Time: {point['time']:.2f}s\n"
            f"Servo: {servo_deg:.1f} deg\n"
            f"Track: {'ON' if ctrl.tracking else 'OFF'}\n"
            f"[Space]: Pause, [T]: Track, [Arrows]: Seek/Speed"
        )

        mj.mjr_overlay(
            mj.mjtFont.mjFONT_NORMAL,
            mj.mjtGridPos.mjGRID_TOPLEFT,
            viewport,
            info,
            "",
            context,
        )

        glfw.swap_buffers(window)
        glfw.poll_events()

    glfw.terminate()


if __name__ == "__main__":
    main()
