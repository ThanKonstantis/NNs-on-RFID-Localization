import numpy as np


def rotate_points_to_zero(x, y, rfid_pos):
    """Rotate a trajectory so it starts at the origin and ends on the positive x-axis.

    Returns the rotated x/y coords, the rotated RFID tag position, and the rotation angle.
    """
    y_new = y - y[0]
    x_new = x - x[0]

    rfid_new = np.array([rfid_pos[0] - x[0], rfid_pos[1] - y[0]])

    angle = np.arctan2(y_new[-1], x_new[-1])

    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                 [-np.sin(angle), np.cos(angle)]])

    rotated_points = np.dot(rotation_matrix, np.vstack((x_new, y_new)))
    x_rot, y_rot = rotated_points[0], rotated_points[1]

    x_rot = np.where(np.abs(x_rot) < 1e-10, 0, x_rot)
    y_rot = np.where(np.abs(y_rot) < 1e-10, 0, y_rot)

    rfid_rot = np.dot(rotation_matrix, rfid_new)

    if rfid_rot[1] < 0:
        y_rot = -y_rot
        rfid_rot[1] = -rfid_rot[1]

    return x_rot, y_rot, rfid_rot, angle


def rotate_with_respect(x_start, y_start, angle, x, y):
    """Rotate a secondary trajectory using the same angle computed from the primary one."""
    rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                 [-np.sin(angle), np.cos(angle)]])

    y_new = y - y_start
    x_new = x - x_start

    rotated_points = np.dot(rotation_matrix, np.vstack((x_new, y_new)))
    x_rot, y_rot = rotated_points[0], rotated_points[1]

    x_rot = np.where(np.abs(x_rot) < 1e-10, 0, x_rot)
    y_rot = np.where(np.abs(y_rot) < 1e-10, 0, y_rot)

    return x_rot, y_rot
