import numpy as np
import matplotlib.pyplot as plt
from numpy._typing import NDArray


def print_figure(figure):
    x = figure[:, 0]
    y = figure[:, 1]

    if figure.ndim == 2:
        plt.plot(x, y)
        plt.title('My 2D figure')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    elif figure.ndim == 3:
        z = figure[:, 2]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        faces = np.array([
            [0, 1, 4],
            [1, 2, 4],
            [2, 3, 4],
            [3, 0, 4],
            [0, 1, 2],
            [0, 2, 3]
        ])

        ax.plot_trisurf(x, y, z, triangles=faces, cmap='viridis', edgecolor='blue', alpha=0.8)

        ax.set_title('My 3D figure')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.show()


def rotation(figure: NDArray[float], rotation_degree, if_counterclockwise) -> NDArray[float]:
    rotation_radians = np.radians(rotation_degree)
    rotational_matrix = np.array([
        [np.cos(rotation_radians), -np.sin(rotation_radians)],
        [np.sin(rotation_radians), np.cos(rotation_radians)]
    ])

    if if_counterclockwise:
        return np.dot(figure, rotational_matrix.T)
    else:
        return np.dot(figure, rotational_matrix)


def scaling(figure, scale):
    scaling_matrix = np.array([
        [scale, 0],
        [0, scale]
    ])

    return np.dot(figure, scaling_matrix)


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
pyramid = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1]
])

print_figure(batman)

rotated_figure = rotation(batman, 90, True)
print_figure(rotated_figure)

scaled_figure = scaling(batman, 2)
print_figure(scaled_figure)
