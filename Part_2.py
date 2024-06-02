import cv2
import numpy as np


batman = np.array([[0, 0], [1, 0.2], [0.4, 1], [0.5, 0.4], [0, 0.8], [-0.5, 0.4], [-0.4, 1], [-1, 0.2], [0, 0]])

resize = 2

scale_factor = 126
offset = np.array([255, 255])
batman_translated = (batman * scale_factor + offset).astype(int)
batman_translated[:, 1] = 509 - batman_translated[:, 1]

image = np.ones((510, 510), dtype=np.uint8) * 255

for i in range(len(batman_translated) - 1):
    # noinspection PyTypeChecker
    cv2.line(image, tuple(batman_translated[i]), tuple(batman_translated[i + 1]), 69, 2)

# noinspection PyTypeChecker
cv2.line(image, (0, 255), (509, 255), 69, 1)
# noinspection PyTypeChecker
cv2.line(image, (255, 0), (255, 509), 69, 1)

rows, cols = image.shape
center = ((image.shape[1] - 1) / 2.0, (image.shape[0] - 1) / 2.0)

M = cv2.getRotationMatrix2D(center, 90, 1)
rotated_batman = cv2.warpAffine(image, M, (cols, rows))

resized_batman = (batman * scale_factor * resize + offset).astype(int)
resized_batman[:, 1] = 509 - resized_batman[:, 1]

image_resize = np.ones((510, 510), dtype=np.uint8) * 255

for i in range(len(resized_batman) - 1):
    # noinspection PyTypeChecker
    cv2.line(image_resize, tuple(resized_batman[i]), tuple(resized_batman[i + 1]), 69, 2)

# noinspection PyTypeChecker
cv2.line(image_resize, (0, 255), (509, 255), 69, 1)
# noinspection PyTypeChecker
cv2.line(image_resize, (255, 0), (255, 509), 69, 1)

# Rotate axis
x_axis = np.array([[0, 255],
                   [509, 255]])
y_ais = np.array([[255, 0],
                  [255, 509]])

rows, cols = x_axis.shape

rotational_matrix = cv2.getRotationMatrix2D(center, 45, 1)
rotated_axis = cv2.transform(x_axis.reshape(-1, 1, 2), rotational_matrix).reshape(-1, 2).astype(int)
rotated_batman_coordinates = cv2.transform(batman_translated.reshape(-1, 1, 2), rotational_matrix).reshape(-1, 2).astype(int)

rotated_batman_coordinates[:, 1] = 509 - rotated_batman_coordinates[:, 1]
image_rotated = np.ones((510, 510), dtype=np.uint8) * 255

for i in range(len(rotated_batman_coordinates) - 1):
    # noinspection PyTypeChecker
    cv2.line(image_rotated, tuple(rotated_batman_coordinates[i]), tuple(rotated_batman_coordinates[i + 1]), 69, 2)

# noinspection PyTypeChecker
cv2.line(image_rotated, tuple(rotated_axis[0]), tuple(rotated_axis[1]), 69, 1)
# noinspection PyTypeChecker
cv2.line(image_rotated, (255, 0), (255, 509), 69, 1)

cv2.imshow("Original Batman", image)
cv2.imshow('Rotated Batman', rotated_batman)
cv2.imshow('Resized Batman', image_resize)
cv2.imshow('Rotated axis', image_rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
