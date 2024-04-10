import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_laplace
def circle(radius, size):
    x, y = np.meshgrid(np.arange(size), np.arange(size))
    distance = np.sqrt((x - size / 2) ** 2 + (y - size / 2) ** 2)
    circle = distance <= radius
    return circle.astype(float)
radius = 50
size = 200
c = circle(radius, size)
plt.imshow(c, cmap='gray')
plt.title('Circle Image')
plt.axis('off')
plt.show()
sigmas = []
gaussian_response = []
laplacian_response = []
for sigma in range(1, 100):
    sigmas.append(sigma)
    gaussian_response.append(gaussian_filter(c.astype(float), sigma)[size//2,size//2])
    laplacian_response.append(sigma**2 * gaussian_laplace(c.astype(float), sigma)[size//2,size//2])
plt.plot(sigmas, gaussian_response, label='Gaussian Response')
plt.plot(sigmas, laplacian_response, label='Laplacian Response')
plt.xlabel('Sigma')
plt.ylabel('Filter Response')
plt.legend()
plt.show()

