import csv
import matplotlib.pyplot as plt

x_positions = []
y_positions = []

# Read the detected marker positions.
with open("data/processed/IMG_5539_movement.csv") as file:
    for row in csv.DictReader(file):
        if row["detected"] == "1":
            x_positions.append(float(row["x_pixels"]))
            y_positions.append(float(row["y_pixels"]))

# Draw the path, marking its beginning and end.
plt.figure(figsize=(6, 6))
plt.plot(x_positions, y_positions, color="blue")
plt.scatter(x_positions[0], y_positions[0], color="green", label="Start")
plt.scatter(x_positions[-1], y_positions[-1], color="red", label="End")

plt.xlabel("Horizontal position (pixels)")
plt.ylabel("Vertical position (pixels)")
plt.title("SSD movement — 2.5 to 7.3 seconds")

# Image coordinates increase downward, so display them that way.
plt.gca().invert_yaxis()
plt.gca().set_aspect("equal", adjustable="box")
plt.legend()
plt.grid()
plt.tight_layout()

plt.savefig("data/processed/IMG_5539_movement.png")
plt.close()
print("Saved data/processed/IMG_5539_movement.png")
