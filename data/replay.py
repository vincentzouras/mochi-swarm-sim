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


# --- Camera Control ---
class CameraController:
    def __init__(self, model, cam, scene):
        self.model = model
        self.cam = cam
        self.scene = scene
        self.lastx = 0
        self.lasty = 0
        self.button_left = False
        self.button_middle = False
        self.button_right = False
        self.tracking = False

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
        if act == glfw.PRESS and key == glfw.KEY_SPACE:
            self.tracking = not self.tracking
            print(f"Camera Tracking: {'ON' if self.tracking else 'OFF'}")


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
            # Strip whitespace from headers
            reader.fieldnames = [name.strip() for name in reader.fieldnames]

            for row in reader:
                try:
                    ts = float(row["timestamp"])
                    pos = [
                        float(row["pos_x"]),
                        float(row["pos_y"]),
                        float(row["pos_z"]),
                    ]

                    # Angles (radians?)
                    pitch = float(row["pitch"])
                    roll = float(row["roll"])
                    yaw = float(row["yaw"])
                    quat = euler_to_quaternion(roll, pitch, yaw)

                    # Servo Angle (in degrees, need to convert to radians)
                    servo_deg = float(row.get("servo_angle", 0.0))
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

    # --- FIND JOINT ADDRESSES ---
    # A. Find the Main Body Free Joint (Position/Rotation)
    body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, BODY_MAIN)
    if body_id == -1:
        print(f"Error: Body '{BODY_MAIN}' not found.")
        return
    jnt_adr = model.body_jntadr[body_id]
    qpos_main_adr = model.jnt_qposadr[jnt_adr]

    # B. Find the Servo Joint (Motor Tilt)
    servo_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, JOINT_SERVO)
    if servo_id == -1:
        print(f"Warning: Joint '{JOINT_SERVO}' not found. Servo will not move.")
        qpos_servo_adr = -1
    else:
        qpos_servo_adr = model.jnt_qposadr[servo_id]

    # 2. Setup Window
    glfw.init()
    window = glfw.create_window(1200, 900, "Mochi Replay", None, None)
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
    ctrl = CameraController(model, cam, scene)
    glfw.set_cursor_pos_callback(window, ctrl.mouse_move)
    glfw.set_mouse_button_callback(window, ctrl.mouse_button)
    glfw.set_scroll_callback(window, ctrl.scroll)
    glfw.set_key_callback(window, ctrl.key_callback)

    # 5. Load Data
    flight_log = load_csv_data(CSV_FILE_PATH)
    if not flight_log:
        print("No valid data loaded.")
        return

    idx = 0
    start_real_time = time.time()
    start_log_time = flight_log[0]["time"]

    while not glfw.window_should_close(window):
        # Loop playback
        if idx >= len(flight_log):
            idx = 0
            start_real_time = time.time()

        # Sync time
        elapsed = time.time() - start_real_time
        while (
            idx < len(flight_log) - 1
            and (flight_log[idx]["time"] - start_log_time) < elapsed
        ):
            idx += 1

        point = flight_log[idx]

        # --- UPDATE PHYSICS STATE ---

        # 1. Update Main Body (Pos + Quat)
        data.qpos[qpos_main_adr : qpos_main_adr + 3] = point["pos"]
        data.qpos[qpos_main_adr + 3 : qpos_main_adr + 7] = point["quat"]

        # 2. Update Servo Angle
        if qpos_servo_adr != -1:
            data.qpos[qpos_servo_adr] = point["servo"]

        # 3. Propagate changes to geometry
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

        # Overlay
        servo_deg = math.degrees(point["servo"])
        info = (
            f"Time: {point['time']:.2f}s\n"
            f"Servo: {servo_deg:.1f} deg\n"
            f"Track: {'ON' if ctrl.tracking else 'OFF'}"
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
