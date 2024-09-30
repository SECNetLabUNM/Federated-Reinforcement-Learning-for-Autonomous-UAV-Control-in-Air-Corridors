# import numpy as np
import open3d as o3d
#
# # Axis-angle representation
# axis = np.array([0, 0, 1])  # Rotate around y-axis
#
# vec2=np.array([1,1,1])
# unit_vec2=vec2/np.linalg.norm(vec2)
# angle = np.arccos(np.dot(axis, unit_vec2))
# rot_vec = np.cross(axis, unit_vec2)
# rot_vec = rot_vec.astype(float)
# rot_vec /= np.linalg.norm(rot_vec)
# # angle = np.pi / 4  # 45 degrees in radians
# # axis_angle = axis * angle
# rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(angle * rot_vec)
#
# print("Rotation Matrix from Axis-Angle:")
# print(rotation_matrix)
# print(np.linalg.norm(rotation_matrix,axis=0))
# print(np.dot(rotation_matrix,np.array([0,0,1])))
# print(np.dot(rotation_matrix,np.array([1,0,0])))
# #### np.dot(rotation_matrix,np.array([0,0,1]))  ----> array([0.57735027, 0.57735027, 0.57735027])
#
# # Yaw, Pitch, Roll
# yaw = np.pi / 4  # 45 degrees
# pitch = 0
# roll =  np.pi / 4
#
# # Create individual rotation matrices
# R_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, 1]) * np.pi/4)
# R_pitch = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * 0)
# R_roll = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * np.pi/4)
#
# # Combine rotations
# combined_rotation_matrix = R_yaw @ R_pitch @ R_roll
# # print(R_yaw  @ R_roll)
# print("\nCombined Rotation Matrix from Yaw, Pitch, Roll:")
# print(combined_rotation_matrix)
# print(np.dot(combined_rotation_matrix,np.array([0,0,1])))
#



# # Yaw, Pitch, Roll
# yaw = np.pi / 4  # 45 degrees
# pitch = 0
# roll =  np.pi / 4
#
# # Create individual rotation matrices
# R_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, 1]) * yaw)
# R_pitch = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * pitch)
# R_roll = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * roll)
#
# # Combine rotations
# combined_rotation_matrix = R_yaw @ R_pitch @ R_roll
# # print(R_yaw  @ R_roll)
# print("\nCombined Rotation Matrix from Yaw, Pitch, Roll:")
# print(combined_rotation_matrix)

import numpy as np
from scipy.spatial.transform import Rotation as R

def normalize_vector(v):
    """ Normalize a vector. """
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2. """
    a, b = normalize_vector(vec1), normalize_vector(vec2)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

    return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))

def euler_from_rotation_matrix(ro):
    """ Convert rotation matrix to Euler angles. """
    r = R.from_matrix(ro)
    return r.as_euler('xyz', degrees=True)

vec1 = np.random.rand(3)
vec1 /= np.linalg.norm(vec1)
vec2 = np.random.rand(3)
vec2 /= np.linalg.norm(vec2)
# Compute rotation matrix
ro= rotation_matrix_from_vectors(vec1, vec2)

# Extract Euler angles
yaw, pitch, roll = euler_from_rotation_matrix(ro)

R_yaw = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 0, 1]) * yaw)
R_pitch = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([1, 0, 0]) * pitch)
R_roll = o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([0, 1, 0]) * roll)

# Combine rotations
combined_rotation_matrix = R_yaw @ R_pitch @ R_roll
print(vec1, vec2)
print(np.dot(combined_rotation_matrix,vec1))
print(np.dot(combined_rotation_matrix,vec2))