import pygame
import math
import csv
import time
import numpy as np
import random 

# Initialize Pygame and set up the display.
pygame.init()
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Ackerman Kinematics - Complex Maze")
clock = pygame.time.Clock()
FPS = 30

# -------------------------------
# Vehicle Class with Ackerman Kinematics
# -------------------------------
class Vehicle:
    def __init__(self, x, y, theta, L=50):
        self.x = x          # x-coordinate
        self.y = y          # y-coordinate
        self.theta = theta  # Orientation (radians)
        self.v = 0.0        # Linear velocity
        self.delta = 0.0    # Steering angle (radians)
        self.L = L          # Wheelbase (distance between axles)
        self.max_steer = math.radians(60)  # Maximum steer (30°)
        self.max_speed = 50.0              # Maximum speed
        self.acceleration = 0.5            # Acceleration increment per frame
        self.deceleration = 1.0            # Deceleration increment per frame

    def update(self, dt):
        # Update vehicle position using Euler integration of the Ackerman equations.
        self.x += self.v * math.cos(self.theta) * dt
        self.y += self.v * math.sin(self.theta) * dt
        self.theta += (self.v / self.L) * math.tan(self.delta) * dt

    def draw(self, surface):
        # Draw the vehicle as a triangle to indicate direction.
        front_length = 20  # Distance from center to front of vehicle
        width = 10         # Half-width of the vehicle
        
        # Compute the triangle points in the vehicle's coordinate frame.
        front_x = self.x + front_length * math.cos(self.theta)
        front_y = self.y + front_length * math.sin(self.theta)
        back_left_x = self.x - width * math.cos(self.theta) - width * math.sin(self.theta)
        back_left_y = self.y - width * math.sin(self.theta) + width * math.cos(self.theta)
        back_right_x = self.x - width * math.cos(self.theta) + width * math.sin(self.theta)
        back_right_y = self.y - width * math.sin(self.theta) - width * math.cos(self.theta)
        
        points = [(front_x, front_y), (back_left_x, back_left_y), (back_right_x, back_right_y)]
        pygame.draw.polygon(surface, (255, 0, 0), points)

class Goal:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
    
    def draw(self, surface):
        pygame.draw.circle(surface, (0, 0, 255), (int(self.x), int(self.y)), self.radius)

# -------------------------------
# Obstacle Class (Circular)
# -------------------------------
class Obstacle:
    def __init__(self, x, y, radius):
        self.x = x
        self.y = y
        self.radius = radius
    

    def generate_circular_obstacles(num_obstacles, field_width, field_height, min_radius=10, max_radius=30, vehicle_pos=(100, 100), goal_pos=(700, 500), avoid_distance=50):
        obstacles = []
        vehicle_x, vehicle_y = vehicle_pos
        goal_x, goal_y = goal_pos

        for _ in range(num_obstacles):
            # Generate a random radius within the specified range
            radius = random.uniform(min_radius, max_radius)
            # Ensure the obstacle is fully within the simulation boundaries.
            # Generate a candidate position for the obstacle's center
            x = random.uniform(radius+50, (field_width - radius)-50)
            y = random.uniform(radius+50, (field_height - radius)-50)
            
            # Re-sample the position if the obstacle is too close to the vehicle.
            # The condition ensures that the distance between the obstacle's center
            # and the vehicle is greater than the sum of the obstacle's radius and the safe avoid_distance.
            while (math.sqrt((x - vehicle_x) ** 2 + (y - vehicle_y) ** 2) <= (avoid_distance + radius)) or (math.sqrt((x - goal_x) ** 2 + (y - goal_y) ** 2) <= (avoid_distance + radius)):
                x = random.uniform(radius, field_width - radius)
                y = random.uniform(radius, field_height - radius)
            obstacles.append(Obstacle(x, y, radius))
            #obstacles.append((x, y, radius))
        return obstacles

    def draw(self, surface):
        pygame.draw.circle(surface, (0, 255, 0), (int(self.x), int(self.y)), self.radius)

# -------------------------------
# Wall Class (Rectangular Obstacles)
# -------------------------------
class Wall:
    def __init__(self, x, y, width, height):
        self.x = x          # Top-left x-coordinate
        self.y = y          # Top-left y-coordinate
        self.width = width
        self.height = height

    def get_edges(self):
        # Returns the four edges (line segments) of the rectangular wall.
        return [
            ((self.x, self.y), (self.x + self.width, self.y)),                      # Top edge
            ((self.x + self.width, self.y), (self.x + self.width, self.y + self.height)),  # Right edge
            ((self.x + self.width, self.y + self.height), (self.x, self.y + self.height)),   # Bottom edge
            ((self.x, self.y + self.height), (self.x, self.y))                        # Left edge
        ]

    def draw(self, surface):
        pygame.draw.rect(surface, (128, 128, 128), pygame.Rect(self.x, self.y, self.width, self.height), 2)

# -------------------------------
# Helper: Ray and Line Segment Intersection
# -------------------------------

class Lidar:

    def ray_line_intersection(self, ray_origin, ray_dir, p1, p2):
        """
        Computes the intersection between a ray and a line segment.
        Returns the distance along the ray where the intersection occurs if valid, else returns None.
        """
        (x0, y0) = ray_origin
        (dx, dy) = ray_dir
        (x1, y1) = p1
        (x2, y2) = p2

        denominator = dx * (y2 - y1) - dy * (x2 - x1)
        if denominator == 0:
            return None  # Lines are parallel.
        
        t = ((x1 - x0) * (y2 - y1) - (y1 - y0) * (x2 - x1)) / denominator
        u = ((x1 - x0) * dy - (y1 - y0) * dx) / denominator
        if t >= 0 and 0 <= u <= 1:
            return t
        return None
    
    def simulate_lidar(self, vehicle, obstacles, num_rays=15, fov=math.radians(120), max_range=500):
        """
        Simulates a LIDAR sensor by casting multiple rays across a field of view (fov)
        and determining the distance to the closest obstacle (walls or circular obstacles).
        Returns lists of distances and ray end-points for visualization.
        """
        angles = np.linspace(-fov/2, fov/2, num_rays)
        distances = []
        ray_points = []
        
        for a in angles:
            ray_angle = vehicle.theta + a
            min_dist = max_range  # Default: nothing is hit within max_range.
            ray_dir = (math.cos(ray_angle), math.sin(ray_angle))
            
            for obs in obstacles:
                if isinstance(obs, Obstacle):
                    # Check intersection with a circle (obstacle).
                    dx = obs.x - vehicle.x
                    dy = obs.y - vehicle.y
                    t_center = dx * math.cos(ray_angle) + dy * math.sin(ray_angle)
                    if t_center > 0:
                        closest_x = vehicle.x + t_center * math.cos(ray_angle)
                        closest_y = vehicle.y + t_center * math.sin(ray_angle)
                        dist_to_center = math.hypot(obs.x - closest_x, obs.y - closest_y)
                        if dist_to_center < obs.radius:
                            dt = math.sqrt(obs.radius**2 - dist_to_center**2)
                            intersection_dist = t_center - dt
                            if intersection_dist < min_dist:
                                min_dist = intersection_dist
                elif isinstance(obs, Wall):
                    # Check intersection with each edge of the wall.
                    for edge in obs.get_edges():
                        t = self.ray_line_intersection((vehicle.x, vehicle.y), ray_dir, edge[0], edge[1])
                        if t is not None and t < min_dist:
                            min_dist = t
            distances.append(min_dist)
            end_x = vehicle.x + min_dist * math.cos(ray_angle)
            end_y = vehicle.y + min_dist * math.sin(ray_angle)
            ray_points.append((end_x, end_y))
            
        return distances, ray_points
