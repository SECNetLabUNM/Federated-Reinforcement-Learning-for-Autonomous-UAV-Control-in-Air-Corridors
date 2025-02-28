import matplotlib.pyplot as plt
import numpy as np


def is_inside_circle(point, circle_radius):
    """Check if a point is inside a circle with given radius."""
    return point[0] ** 2 + point[1] ** 2 <= circle_radius ** 2


def rotate_point(point, angle):
    """Rotate a point by a given angle around the origin."""
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return rotation_matrix.dot(point)


def generate_hexagon_grid(circle_radius, side_length):
    """Generate a grid of hexagon centers and vertices within a given circle."""
    hexagon_centers = []
    hexagon_vertex_list = []

    # 60 degrees in radians and vertical spacing between centers
    angle = np.deg2rad(60)
    vertical_spacing = side_length * np.sqrt(3)

    # Determine the range for the grid
    grid_range_x = int(circle_radius / side_length) + 1
    grid_range_y = int(circle_radius / (vertical_spacing / 2)) + 1

    # Generate potential hexagon centers and their vertices
    for i in range(-grid_range_y, grid_range_y + 1):
        for j in range(-grid_range_x, grid_range_x + 1):
            # Offset for even and odd rows
            offset = 0 if i % 2 == 0 else side_length * 1.5
            center_x = j * 3 * side_length + offset
            center_y = i * vertical_spacing / 2
            center = (center_x, center_y)

            # Add the center if it's inside the circle
            if is_inside_circle(center, circle_radius):
                hexagon_centers.append(center)

            # Calculate vertices and add those that are inside the circle
            hex_vertices = [
                (center[0] + np.cos(k * angle) * side_length, center[1] + np.sin(k * angle) * side_length)
                for k in range(6)
            ]
            for vertex in hex_vertices:
                if is_inside_circle(vertex, circle_radius) and vertex not in hexagon_vertex_list:
                    can_add_vertex = True
                    for exist in hexagon_vertex_list:
                        if np.linalg.norm(np.array(vertex) - np.array(exist)) < (side_length / 2):
                            can_add_vertex = False
                            break
                    if can_add_vertex:
                        hexagon_vertex_list.append(vertex)

    return hexagon_centers, hexagon_vertex_list


# Parameters
circle_radius = 2
hexagon_side_length = 0.60  # Now with a side length of 0..5

# Generate the hexagon grid
centers, vertices = generate_hexagon_grid(circle_radius, hexagon_side_length)

# Choose a random angle between 0 and 360 degrees (in radians)
random_angle = 0  # np.deg2rad(random.uniform(0, 360))

# Apply rotation to each center and vertex
rotated_centers = [rotate_point(center, random_angle) for center in centers]
rotated_vertices = [rotate_point(vertex, random_angle) for vertex in vertices]

# Plot the circle and the rotated points
plt.figure(figsize=(6, 6))
circle = plt.Circle((0, 0), circle_radius, color='blue', fill=False)
plt.gca().add_patch(circle)

# Plotting the rotated hexagon centers
rotated_center_x_vals, rotated_center_y_vals = zip(*rotated_centers)
# plt.scatter(rotated_center_x_vals, rotated_center_y_vals, color='green', label='Rotated Hexagon Centers')
plt.scatter(rotated_center_x_vals, rotated_center_y_vals, color='red')
# Plotting the rotated vertices
rotated_vertex_x_vals, rotated_vertex_y_vals = zip(*rotated_vertices)

# plt.scatter(rotated_vertex_x_vals, rotated_vertex_y_vals, color='red', label='Rotated Vertices')
plt.scatter(rotated_vertex_x_vals, rotated_vertex_y_vals, color='red')
print(len(rotated_vertex_x_vals) + len(rotated_center_x_vals))
plt.xlim(-circle_radius, circle_radius)
plt.ylim(-circle_radius, circle_radius)
plt.gca().set_aspect('equal', adjustable='box')
plt.title("Hexagon Vertices Inside a Releasing Circle Plane")
plt.savefig('hexagon_release.jpg')
plt.savefig('hexagon_release.pdf')
# plt.legend()
plt.show()
