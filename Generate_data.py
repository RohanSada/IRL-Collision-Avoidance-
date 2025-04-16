import pygame
import math
import csv
import time
import numpy as np
import random 
from SimClass import *

goal = Goal(700, 500, 20)

circle_obstacles = Obstacle.generate_circular_obstacles(20, 750, 500)

maze_walls = [
    # Outer boundaries
    Wall(50, 50, 700, 10),    # Top wall
    Wall(50, 50, 10, 500),    # Left wall
    Wall(50, 540, 700, 10),   # Bottom wall
    Wall(740, 50, 10, 500),   # Right wall
]

obstacles = circle_obstacles + maze_walls

vehicle = Vehicle(100, 100, math.radians(0), L=50)

demonstration_data = []
start_time = time.time()

lidar = Lidar()

# -------------------------------
# Main Simulation Loop
# -------------------------------
running = True
while running:
    dt = clock.get_time() / 1000.0  # Convert milliseconds to seconds.
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Process keyboard input for acceleration and steering.
    keys = pygame.key.get_pressed()
    if keys[pygame.K_UP]:
        vehicle.v = min(vehicle.v + vehicle.acceleration, vehicle.max_speed)
    elif keys[pygame.K_DOWN]:
        vehicle.v = max(vehicle.v - vehicle.deceleration, 0)
    else:
        vehicle.v *= 0.99  # Apply friction when no key is pressed.

    if keys[pygame.K_LEFT]:
        vehicle.delta = max(vehicle.delta - math.radians(1), -vehicle.max_steer)
    elif keys[pygame.K_RIGHT]:
        vehicle.delta = min(vehicle.delta + math.radians(1), vehicle.max_steer)
    else:
        vehicle.delta *= 0.9  # Gradually return steering to zero.
    
    # Update the vehicle state.
    vehicle.update(dt)
    
    # Simulate the LIDAR sensor.
    lidar_distances, ray_points = lidar.simulate_lidar(vehicle, obstacles)
    
    # Obtains direct distance between the vehicle and the goal 
    goal_distance = math.sqrt((vehicle.x-goal.x)**2 + (vehicle.y-goal.y)**2)

    # Log demonstration data.
    current_time = time.time() - start_time
    demonstration_entry = {
        "time": current_time,
        "x": vehicle.x,
        "y": vehicle.y,
        "theta": vehicle.theta,
        "v": vehicle.v,
        "delta": vehicle.delta,
        "lidar": lidar_distances,
        "GoalDistance": goal_distance
    }
    demonstration_data.append(demonstration_entry)
    
    # Rendering
    screen.fill((255, 255, 255))  # Clear screen with white background.
    for obs in obstacles:
        obs.draw(screen)
    goal.draw(screen)
    vehicle.draw(screen)
    for p in ray_points:
        pygame.draw.line(screen, (0, 0, 255), (vehicle.x, vehicle.y), p, 1)
    
    pygame.display.flip()
    clock.tick(FPS)

# -------------------------------
# Save Demonstration Data to CSV (Optional)
# -------------------------------
with open("./Data/demonstration_data.csv", "w", newline="") as csvfile:
    fieldnames = ["time", "x", "y", "theta", "v", "delta", "lidar", "GoalDistance"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for entry in demonstration_data:
        # Convert LIDAR readings list to a string for CSV storage.
        entry["lidar"] = str(entry["lidar"])
        writer.writerow(entry)

pygame.quit()