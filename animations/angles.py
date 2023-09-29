import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


# Function to generate random angles between 0 and 2π
def generate_random_angles():
    return np.random.uniform(0, 2 * np.pi, 4)


# Create the initial plot with dials and bar plot
num_rotations = 4
fig, axs = plt.subplots(2, num_rotations, figsize=(12, 6))
fig.suptitle("Quantum Random Rotations Angles For 4 Qubits.")

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.2)  # Decreased from 0.5 to 0.2

# Initialize arrays to store the arrow and text artists
arrows = []
angle_texts = []

# Initialize an array to store the accumulated angles
accumulated_angles = []

# Initialize the bar plots
bar_axes = axs[1, :]


# Function to update the plot with new angles
def update(frame):
    random_angles = generate_random_angles()
    accumulated_angles.extend(random_angles)  # Accumulate the random angles

    for i, ax in enumerate(axs[0]):
        ax.clear()
        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect('equal', adjustable='datalim')
        ax.set_facecolor('white')  # Set white background
        ax.set_xticks([])
        ax.set_yticks([])

        # Add a small Qubit indicating the dial position
        circle = plt.Circle((0, 0), 0.2, fill=True, color='orange')
        ax.add_artist(circle)

        # Update the arrow indicating the angle of rotation
        arrow = plt.Arrow(0, 0, 0.18 * np.cos(random_angles[i]), 0.18 * np.sin(random_angles[i]), width=0.02,
                          color='black')
        ax.add_artist(arrow)
        arrows.append(arrow)

        # Add text indicating the angle in radians
        angle_text = ax.text(0, -0.4, f'{random_angles[i]:.2f} rad', color='black', ha='center')
        angle_texts.append(angle_text)

    # Update the angle distribution plots for each circle
    for i, bar_ax in enumerate(bar_axes):
        bar_ax.clear()
        bar_ax.hist(accumulated_angles[i::num_rotations], bins=20, color='blue', alpha=0.7)
        bar_ax.set_title(f"Angle Distribution Qubit {i + 1}")
        bar_ax.set_xlabel("Angle (radians)")
        bar_ax.set_ylabel("Frequency")


# Create the animation
ani = FuncAnimation(fig, update, frames=1000, interval=100)  # Update every 1000 milliseconds (1 second), 5000 frames

# Save the animation
ani.save('../../figs/angle_animation.mp4', dpi=240, bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])

plt.show()
