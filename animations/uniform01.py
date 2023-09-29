import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import norm
from skimage import io, img_as_float

# Number of samples
n_samples = 600

# Uniform distribution samples
uniform_samples = np.random.randint(0, 16, n_samples)

# Marsaglia polar method to transform uniform to Gaussian
gaussian_samples = []
while len(gaussian_samples) < n_samples:
    u1, u2 = np.random.uniform(-1, 1, 2)
    s = u1**2 + u2**2
    if 0 < s < 1:
        gaussian_samples.append(u1 * np.sqrt(-2 * np.log(s) / s))
        gaussian_samples.append(u2 * np.sqrt(-2 * np.log(s) / s))

gaussian_samples = np.array(gaussian_samples[:n_samples])

# Load an image
image_path = 'galtonBoard.png' # Change this to your image path
image = img_as_float(io.imread(image_path, as_gray=True))

# Noise accumulation
accumulated_noise = np.zeros_like(image)

# Initialize figure and axes
fig, (ax1, (ax2, ax4)) = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [1, 1], 'width_ratios': [1, 1]})
ax1 = plt.subplot(2, 1, 1)
ax2 = plt.subplot(2, 2, 3)
ax4 = plt.subplot(2, 2, 4)

# Gaussian parameters
mean = 0
std_dev = 1
x = np.linspace(mean - 5*std_dev, mean + 5*std_dev, 1000)

# Qubit labels for uniform distribution
qubit_labels = [f"|{format(i, '04b')}> " for i in range(16)]

# Different shades of gray
gray_shades = [str(i/16) for i in range(16)]

# Animation function
def animate(i):
    global accumulated_noise # Declare it as global to modify it inside the function

    ax1.clear()
    ax2.clear()
    ax4.clear()
    fig.suptitle(f'Current Sample: {i + 1}', fontsize=16)
    # Plotting Uniform Distribution
    counts, _, patches = ax1.hist(uniform_samples[:i], bins=16, density=True, alpha=0.5)
    ax1.set_title('Uniform Distribution Via Superposition')
    ax1.set_xticks(np.arange(16))
    ax1.set_xticklabels(qubit_labels, rotation=45, fontsize=8)
    for patch, shade in zip(patches, gray_shades):
        patch.set_facecolor(shade)

    # Plotting Gaussian Distribution
    ax2.hist(gaussian_samples[:i], bins=16, density=True, color='gray', alpha=0.5)

    # Plot the Gaussian curve
    ax2.plot(x, norm.pdf(x, mean, std_dev), 'gray', lw=2)

    # Add the Gaussian normal equation
    ax2.text(mean, 0.3, r'$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^2}$', fontsize=12)

    # Draw vertical lines for 3 standard deviations
    for j in range(-3, 4):
        ax2.axvline(mean + j*std_dev, color='red', linestyle='--')

    ax2.set_title('Gaussian Distribution')

    
    # Add Gaussian noise to the image
    noise = gaussian_samples[i] * np.random.randn(*image.shape)
    accumulated_noise += noise
    noisy_image = image + accumulated_noise
    noisy_image = np.clip(noisy_image, 0, 1)

    # Display the noisy image
    ax4.imshow(noisy_image, cmap='gray')
    ax4.axis('off')
    ax4.set_title('Quantum Noise')

# Create animation
ani = animation.FuncAnimation(fig, animate, frames=n_samples, interval=10, repeat=False)
# Add this before showing the animation

# Then replace plt.show() with:

# plt.close()
# plt.show()
ani.save('../../figs/gaussian_animation.mp4', dpi=240, bitrate=-1, codec="libx264", extra_args=['-pix_fmt', 'yuv420p'])