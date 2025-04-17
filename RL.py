import gymnasium as gym
import numpy as np
import math
import random
from gymnasium import spaces
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from numba import njit

@njit
def cast_ray_numba(x, y, beam_angle, obstacles, obstacle_radius, max_range, step_size):
    d = 0.0
    while d < max_range:
        tx = x + d * math.cos(beam_angle)
        ty = y + d * math.sin(beam_angle)
        for i in range(obstacles.shape[0]):
            dx = tx - obstacles[i, 0]
            dy = ty - obstacles[i, 1]
            if dx*dx + dy*dy <= obstacle_radius*obstacle_radius:
                return d
        d += step_size
    return max_range

@njit
def check_collision_numba(x, y, obstacles, obstacle_radius, vehicle_radius, xmin, xmax, ymin, ymax):
    # Obstacle collisions
    for i in range(obstacles.shape[0]):
        dx = x - obstacles[i, 0]
        dy = y - obstacles[i, 1]
        if dx*dx + dy*dy <= (obstacle_radius + vehicle_radius) ** 2:
            return True
    # World bounds
    if x < xmin or x > xmax or y < ymin or y > ymax:
        return True
    return False

class AckermannVehicleEnv(gym.Env):
    """
    Custom Gym environment for an Ackermann vehicle.
    
    The vehicle:
      - Starts at a fixed point.
      - Has a fixed goal location.
      - Must avoid randomly generated circular obstacles.
      - Uses simulated Lidar sensor beams for obstacle distance readings.
    
    The vehicle dynamics are approximated with a simple Ackermann (bicycle) model.
    """
    metadata = {'render.modes': ['human']}
    
    def __init__(self,
                 start_pos=(0.0, 0.0),
                 start_heading=0.0,
                 goal_pos=(20.0, 20.0),
                 num_obstacles=10,
                 obstacle_radius=1.0,
                 world_bounds=(-50.0, 50.0, -50.0, 50.0),
                 num_lidar_beams=9,
                 lidar_max_range=30.0,
                 dt=0.1,
                 max_episode_steps=500,
                 vehicle_length=2.5):
        super(AckermannVehicleEnv, self).__init__()
        
        # Environment settings
        self.start_pos = start_pos
        self.start_heading = start_heading  # in radians
        self.goal_pos = np.array(goal_pos)
        self.num_obstacles = num_obstacles
        self.obstacle_radius = obstacle_radius
        self.world_bounds = world_bounds  # Format: (xmin, xmax, ymin, ymax)
        self.num_lidar_beams = num_lidar_beams
        self.lidar_max_range = lidar_max_range
        self.dt = dt
        self.max_episode_steps = max_episode_steps
        self.vehicle_length = vehicle_length
        
        xmin, xmax, ymin, ymax = self.world_bounds
        dist_max = math.hypot(xmax-xmin, ymax-ymin)

        # For collision checking, approximate the vehicle as a circle.
        self.vehicle_radius = 1.0
        
        # Action space:
        #   - First value: acceleration (m/s^2) (clipped to [-3, 3])
        #   - Second value: steering angle in radians (clipped to ±30 degrees)
        self.action_space = spaces.Box(low=np.array([-3.0, -np.radians(30)]), 
                                       high=np.array([3.0, np.radians(30)]),
                                       dtype=np.float32)
        
        # Observation space:
        # The observation contains the vehicle state (x, y, heading, velocity) followed by Lidar distances.
        # Bounds:
        #   - x, y: within the world bounds.
        #   - heading: in [-π, π]
        #   - velocity: assumed clipped between -10 and 10 m/s.
        #   - each Lidar beam: distance in [0, lidar_max_range]
        obs_low = np.array([world_bounds[0], world_bounds[2], -np.pi, -10.0, 0.0, -np.pi] + [0.0] * num_lidar_beams, dtype=np.float32)
        obs_high = np.array([world_bounds[1], world_bounds[3], np.pi, 10.0, dist_max, np.pi] + [lidar_max_range] * num_lidar_beams, dtype=np.float32)
        
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)
        
        # Generate obstacles (their centers) randomly in the world
        self.obstacles = []
        self._generate_obstacles()
        self.obstacles = np.array(self.obstacles, dtype=np.float32)
        
        # Initialize vehicle state and episode counter.
        self.reset()
        self.current_step = 0

    def seed(self, seed=None):
        random.seed(seed)
        np.random.seed(seed)
        return [seed]

    def _generate_obstacles(self):
        """
        Populate the environment with randomly placed circular obstacles.
        Ensures obstacles are not placed too close to the start or goal.
        """
        self.obstacles = []
        xmin, xmax, ymin, ymax = self.world_bounds
        for i in range(self.num_obstacles):
            while True:
                x = random.uniform(xmin, xmax)
                y = random.uniform(ymin, ymax)
                point = np.array([x, y])
                # Avoid obstacles too near the start or goal (within 5 units).
                if np.linalg.norm(point - np.array(self.start_pos)) < 5.0 or np.linalg.norm(point - self.goal_pos) < 5.0:
                    continue
                self.obstacles.append(np.array([x, y]))
                break

    def _randomize_goal(self):
        pass
        """
        Randomize the goal position within bounds, not too close to start.
        """
        xmin, xmax, ymin, ymax = self.world_bounds
        while True:
            gx = random.uniform(xmin, xmax)
            gy = random.uniform(ymin, ymax)
            goal = np.array([gx, gy])
            if np.linalg.norm(goal - self.start_pos) < 5.0:
                continue
            self.goal_pos = goal
            break

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self._randomize_goal()
        self._generate_obstacles()
        self.obstacles = np.array(self.obstacles, dtype=np.float32)
        self.vehicle_pos = np.array(self.start_pos, dtype=np.float32)
        self.vehicle_heading = self.start_heading
        self.vehicle_velocity = 0.0
        self.vehicle_steering = 0.0
        self.current_step = 0
        self.prev_dist = np.linalg.norm(self.goal_pos - self.vehicle_pos)
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        # Clip the action to ensure it's within the action space.
        action = np.clip(action, self.action_space.low, self.action_space.high)
        acceleration = action[0]
        steering_angle = action[1]

        # Update vehicle dynamics using a simple bicycle model.
        self.vehicle_velocity += acceleration * self.dt
        self.vehicle_velocity = np.clip(self.vehicle_velocity, -10.0, 10.0)
        self.vehicle_heading += (self.vehicle_velocity / self.vehicle_length) * math.tan(steering_angle) * self.dt
        self.vehicle_heading = (self.vehicle_heading + math.pi) % (2*math.pi) - math.pi
        self.vehicle_pos[0] += self.vehicle_velocity * math.cos(self.vehicle_heading) * self.dt
        self.vehicle_pos[1] += self.vehicle_velocity * math.sin(self.vehicle_heading) * self.dt
        self.vehicle_steering = steering_angle

        dx = self.goal_pos[0] - self.vehicle_pos[0]
        dy = self.goal_pos[1] - self.vehicle_pos[1]
        bearing_to_goal = math.atan2(dy, dx) - self.vehicle_heading
        bearing_to_goal = (bearing_to_goal + math.pi) % (2*math.pi) - math.pi

        dist_to_goal = np.linalg.norm(self.goal_pos - self.vehicle_pos)
        reward = (self.prev_dist - dist_to_goal)*50.0

        self.prev_dist = dist_to_goal

        reward += math.cos(bearing_to_goal) * 0.5

        collision = self._check_collision()

        goal_reached = dist_to_goal < 10.0

        if collision:
            reward -= 100.0
        if goal_reached:
            reward += 500.0

        time_penalty = 0.1
        reward -= time_penalty

        # Increment step counter.
        self.current_step += 1

        # Construct observation.
        obs = self._get_obs()

        # Determine terminated and truncated flags:
        # Ensure they are plain Python booleans.
        terminated = bool(collision or goal_reached)
        truncated = bool((self.current_step >= self.max_episode_steps) and not terminated)

        info = {'collision': collision, 'goal_reached': goal_reached, 'distance_to_goal': dist_to_goal}

        # Return the five values as required by Gymnasium.
        return obs, reward, terminated, truncated, info

    def _get_obs(self):
        """
        Build the observation from the current vehicle state and Lidar sensor data.
        """
        # Vehicle state: [x, y, heading, velocity]
        # 1) vector to goal
        dx = self.goal_pos[0] - self.vehicle_pos[0]
        dy = self.goal_pos[1] - self.vehicle_pos[1]
        dist_to_goal = math.hypot(dx, dy)
        # wrap bearing to [-pi,pi]
        bearing_to_goal = math.atan2(dy, dx) - self.vehicle_heading
        bearing_to_goal = (bearing_to_goal + math.pi) % (2*math.pi) - math.pi
        # Append Lidar readings.
        lidar_readings = self._simulate_lidar()
        state = [self.vehicle_pos[0], self.vehicle_pos[1], self.vehicle_heading, self.vehicle_velocity, dist_to_goal, bearing_to_goal]
        return np.array(state + lidar_readings, dtype=np.float32)

    '''
    def _simulate_lidar(self):
        """
        Simulate Lidar sensor data by emitting beams over a fixed angular range.
        
        Returns a list of distances (one for each beam) at which an obstacle is detected.
        If no obstacle is detected within the maximum range, the reading is set to lidar_max_range.
        """
        readings = []
        # Distribute beams evenly over an angular span (e.g., ±45° relative to vehicle heading)
        angle_span = math.radians(90)
        start_angle = self.vehicle_heading - angle_span / 2.0
        angle_increment = angle_span / (self.num_lidar_beams - 1)
        
        for i in range(self.num_lidar_beams):
            beam_angle = start_angle + i * angle_increment
            distance = self._cast_ray(beam_angle)
            readings.append(distance)
        return readings
    '''
    def _simulate_lidar(self):
        """
        Use the JIT cast_ray_numba for each beam for speed.
        """
        readings = np.empty(self.num_lidar_beams, dtype=np.float32)
        angle_span = math.radians(90)
        start_angle = self.vehicle_heading - angle_span/2.0
        increment = angle_span / (self.num_lidar_beams - 1)
        for i in range(self.num_lidar_beams):
            beam_angle = start_angle + i * increment
            readings[i] = cast_ray_numba(
                self.vehicle_pos[0], self.vehicle_pos[1],
                beam_angle,
                self.obstacles,            # must be a NumPy array of shape (N,2)
                self.obstacle_radius,
                self.lidar_max_range,
                0.5                        # your lidar step size
            )
        return readings.tolist()

    
    def _cast_ray(self, beam_angle):
        """
        Cast a single Lidar ray from the vehicle in the given direction.
        
        Uses a simple incremental search to determine the distance at which an obstacle is encountered.
        """
        step_size = 0.5  # Resolution of the ray-casting in meters.
        distance = 0.0
        while distance < self.lidar_max_range:
            test_x = self.vehicle_pos[0] + distance * math.cos(beam_angle)
            test_y = self.vehicle_pos[1] + distance * math.sin(beam_angle)
            test_point = np.array([test_x, test_y])
            # Check if any obstacle is hit by this beam.
            for obs in self.obstacles:
                if np.linalg.norm(test_point - obs) <= self.obstacle_radius:
                    return distance
            distance += step_size
        return self.lidar_max_range
    
    '''
    def _check_collision(self):
        """
        Check if the vehicle collides with any obstacle or exceeds the world boundaries.
        A collision is detected when the distance between the vehicle and an obstacle center
        is less than the sum of their radii.
        """
        for obs in self.obstacles:
            if np.linalg.norm(self.vehicle_pos - obs) <= (self.obstacle_radius + self.vehicle_radius):
                return True
        xmin, xmax, ymin, ymax = self.world_bounds
        if not (xmin <= self.vehicle_pos[0] <= xmax and ymin <= self.vehicle_pos[1] <= ymax):
            return True
        return False
    '''
    def _check_collision(self):
        """
        Use the JIT check_collision_numba for all collision logic.
        """
        x, y = self.vehicle_pos
        xmin, xmax, ymin, ymax = self.world_bounds
        return bool(check_collision_numba(
            x, y,
            self.obstacles,             # must be a NumPy array of shape (N,2)
            self.obstacle_radius,
            self.vehicle_radius,
            xmin, xmax, ymin, ymax
        ))
    
    def render(self, mode='human'):
        print("Rendering....")
        # Enable interactive plotting
        plt.ion()
        print("Interactive plotting enabled")
        
        # Create figure and axis if not already done.
        if not hasattr(self, '_fig'):
            self._fig, self._ax = plt.subplots(figsize=(6, 6))
        
        # Clear the previous drawings.
        self._ax.clear()
        
        # Draw obstacles (red circles).
        for obs in self.obstacles:
            circle = plt.Circle((obs[0], obs[1]), self.obstacle_radius, color='red', alpha=0.6)
            self._ax.add_artist(circle)
        
        # Draw vehicle as a blue circle.
        vehicle_circle = plt.Circle((self.vehicle_pos[0], self.vehicle_pos[1]),
                                    self.vehicle_radius, color='blue')
        self._ax.add_artist(vehicle_circle)
        
        # Draw a line for the vehicle heading.
        heading_length = self.vehicle_radius * 2.0
        head_x = self.vehicle_pos[0] + heading_length * math.cos(self.vehicle_heading)
        head_y = self.vehicle_pos[1] + heading_length * math.sin(self.vehicle_heading)
        self._ax.plot([self.vehicle_pos[0], head_x], [self.vehicle_pos[1], head_y], color='black')
        
        # Draw goal as a green star.
        self._ax.plot(self.goal_pos[0], self.goal_pos[1], marker='*', markersize=15, color='green')
        
        # Set axis limits based on world bounds.
        xmin, xmax, ymin, ymax = self.world_bounds
        self._ax.set_xlim(xmin, xmax)
        self._ax.set_ylim(ymin, ymax)
        
        # Set title with the current step.
        self._ax.set_title(f"Step: {self.current_step}")
        
        # (Optional) Draw Lidar beams.
        angle_span = math.radians(90)
        start_angle = self.vehicle_heading - angle_span / 2.0
        angle_increment = angle_span / (self.num_lidar_beams - 1)
        for i in range(self.num_lidar_beams):
            beam_angle = start_angle + i * angle_increment
            distance = self._cast_ray(beam_angle)
            end_x = self.vehicle_pos[0] + distance * math.cos(beam_angle)
            end_y = self.vehicle_pos[1] + distance * math.sin(beam_angle)
            self._ax.plot([self.vehicle_pos[0], end_x], [self.vehicle_pos[1], end_y],
                        color='gray', linestyle='--', linewidth=0.5)
        
        # Force drawing and update.
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()
        plt.pause(0.001)
        
    def close(self):
        pass

# ===========================
# Training using Stable Baselines 3
# ===========================
if __name__ == "__main__":
    import argparse
    import os
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import torch

    # Parse command-line arguments.
    parser = argparse.ArgumentParser(description="Train Ackermann Vehicle RL Agent")
    parser.add_argument('--timesteps', type=int, default=1000000, 
                        help="Total number of training timesteps")
    parser.add_argument('--model_path', type=str, default="ackermann_vehicle_ppo_model",
                        help="Path to save the trained model")
    args = parser.parse_args()

    # Create an instance of the environment.
    env = AckermannVehicleEnv()
    print("Env created")

    # Optional: Check that the environment adheres to Gym API standards.
    check_env(env, warn=True)

    vec_env  = DummyVecEnv([lambda: env])
    env = VecNormalize(vec_env, norm_obs=True, norm_reward=False)

    policy_kwargs = dict(# Shared network trunk of two 256‑unit layers
        net_arch=[256, 256],
        activation_fn=torch.nn.ReLU
    )

    # Initialize the PPO model with an MLP policy.
    model = PPO("MlpPolicy", env, verbose=1, policy_kwargs=policy_kwargs)
    print("Model Initialized")

    #render_callback = RenderCallback(render_freq=100)

    # Train the agent.
    print("Training the model...")
    #model.learn(total_timesteps=args.timesteps, callback=render_callback)
    model.learn(total_timesteps=args.timesteps)
    
    # Save the trained model.
    model.save(args.model_path)
    print(f"Model saved as {args.model_path}")