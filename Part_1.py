import numpy as np
import matplotlib.pyplot as plt
from numpy._typing import NDArray


def print_figure(figure):
    x = figure[:, 0]
    y = figure[:, 1]
    if figure.ndim == 2:
        plt.plot([0, 1], [0, 0], color="blue")
        plt.plot([0, 0], [0, 1], color="blue")
        plt.plot(x, y, color="black")
        plt.title('My 2D figure')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    elif figure.ndim == 3:
        plt.plot([0, 0, 1], [0, 0, 0], [0, 0, 0], color="blue")
        plt.plot([0, 0, 0], [0, 0, 1], [0, 0, 0], color="blue")
        plt.plot([0, 0, 0], [0, 0, 0], [0, 0, 1], color="blue")
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

        ax.plot_trisurf(x, y, z, triangles=faces, cmap='viridis', edgecolor='red', alpha=0.8)

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
        resulted_matrix = []
        for vector in figure:
            rotated_vector = np.dot(rotational_matrix, vector)
            resulted_matrix.append(rotated_vector)

        return np.array(resulted_matrix)
    else:
        return np.dot(figure, rotational_matrix)


def scaling(figure, scale):
    scaling_matrix = np.array([
        [scale, 0],
        [0, scale]
    ])

    return np.dot(figure, scaling_matrix)


def reflection(figure, axis):
    transformation_matrix = np.array([
        [1, 0],
        [0, -1]
    ])
    if axis == "x":
        pass
    elif axis == "y":
        transformation_matrix *= -1

    result = []
    for vector in figure:
        rotated_vector = np.dot(transformation_matrix, vector)
        result.append(rotated_vector)

    return np.array(result)


def print_axis(axis, axis_name, figure: NDArray[float] = 0):
    coordinate_system = np.array([])
    if axis.ndim == 1:
        if axis_name == "x":
            plt.plot([0, 0], [0, 1], color="blue")
            coordinate_system = np.array([[axis[0], 0], [0, 1]])
        elif axis_name == "y":
            plt.plot([0, 1], [0, 0], color="blue")
            coordinate_system = np.array([[1, 0], [0, axis[1]]])

        plt.plot([0, axis[0]], [0, axis[1]], color="red")

        new_figure = np.dot(figure, coordinate_system)
        x = new_figure[:, 0]
        y = new_figure[:, 1]
        plt.plot(x, y, color="black")

        plt.title('My 2D figure')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.grid(True)
        plt.show()

    elif axis.ndim == 2:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if axis_name == "x":
            ax.plot([0, 0], [0, 1], [0, 0], color="blue")
            ax.plot([0, 0], [0, 0], [0, 1], color="blue")
        elif axis_name == "y":
            ax.plot([0, 1], [0, 0], [0, 0], color="blue")
            ax.plot([0, 0], [0, 0], [0, 1], color="blue")
        else:
            ax.plot([0, 1], [0, 0], [0, 0], color="blue")
            ax.plot([0, 0], [0, 1], [0, 0], color="blue")

        ax.plot([0, axis[0]], [0, axis[1]], [0, axis[2]], color="red")
        ax.set_title('My 3D figure')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.grid(True)


def to_transform(transformational_matrix, origin):
    result = []
    for vector in origin:
        rotated_vector = np.dot(transformational_matrix, vector)
        result.append(rotated_vector)

    return np.array(result)


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])
pyramid = np.array([
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
    [0.5, 0.5, 1]
])

print_figure(batman)
#
# rotated_figure = rotation(batman, 90, True)
# print_figure(rotated_figure)
#
# scaled_figure = scaling(batman, 2)
# print_figure(scaled_figure)
#
# mirrored_figure = reflection(batman, "y")
# print_figure(mirrored_figure)

axis_to_transform = np.array([1, 0])
rotated_axis = rotation(axis_to_transform, 45, False)
print_axis(rotated_axis, "x", batman)
