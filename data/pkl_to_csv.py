import pickle
import csv

input_pkl = "data/in_circle_data_6.pkl"
output_csv = "data/output.csv"

# Load PKL data
with open(input_pkl, "rb") as f:
    data = pickle.load(f)

rows = []

for entry in data:
    timestamp, sensor_array, mocap_array = entry

    # Mocap positions
    pos_x = mocap_array[0]
    pos_y = mocap_array[1]
    pos_z = mocap_array[2]

    # Sensor orientation
    pitch = sensor_array[1]  # pitch_body
    roll = -sensor_array[2]  # sensor stores -roll_body → fix it
    yaw = sensor_array[3]  # yaw_body

    servo_angle = sensor_array[12]  # servo_angle_deg

    rows.append([timestamp, pos_x, pos_y, pos_z, pitch, roll, yaw, servo_angle])

# Write to CSV
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        ["timestamp", "pos_x", "pos_y", "pos_z", "pitch", "roll", "yaw", "servo_angle"]
    )
    writer.writerows(rows)

print("CSV saved as", output_csv)
