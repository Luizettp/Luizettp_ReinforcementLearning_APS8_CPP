from typing import Optional
import os
import numpy as np
import gymnasium as gym

import pygame

#
# Coverage Path Planning (CPP) environment based on GridWorld with obstacles.
#
# The agent must visit as many free cells as possible while avoiding obstacles.
# The reward function is designed to encourage exploration of new cells and
# discourage revisiting already-visited cells.
#
# Reward function (inspired by deep RL approaches to patrolling/coverage problems):
#   - +1.0 for visiting a new (unvisited) cell
#   - -0.3 for revisiting an already-visited cell
#   - -0.1 step penalty to encourage efficiency
#   - +10.0 bonus for achieving full coverage (all free cells visited)
#   - -5.0 penalty when max steps reached without full coverage
#
# The observation space includes:
#   - Agent's (x, y) location (normalized)
#   - Coverage ratio (proportion of free cells visited)
#   - A 3x3 matrix of neighboring cells centered on the agent,
#     where (1,1) is the agent's position and each cell is:
#       0 = free (not yet visited), 1 = obstacle or wall (including out-of-bounds),
#       2 = already visited position.
#     Cells outside the grid boundaries are treated as walls (1).
#
# The episode ends when all free cells are visited or max steps is reached.
#

class GridWorldCPPEnv(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, size: int = 5, obs_quantity: int = 3, max_steps: int = 200):
        self.size = size
        self.window_size = 512
        self.obs_quantity = obs_quantity
        self.obstacles_locations = []
        self.count_steps = 0
        self.max_steps = max_steps
        self.last_action_normalized = 0.0
        self.last_action = -1
        self.previous_action = -1
        self.last_new_action = -1
        self.last_collision = 0.0
        self.last_new_cell = 0.0
        self.steps_since_new_cell = 0
        self.recent_new_cells = []
        self.recent_collisions = []
        self.recent_actions = []
        self.recent_action_new_cells = []
        self.current_cell_visits = {}
        self.observed_unvisited_cells = set()
        self.memory_dx = 0.0
        self.memory_dy = 0.0
        self.memory_strength = 0.0
        self.memory_age = 1.0
        self.last_observed_opportunity = 0.0
        self.memory_decay = float(os.environ.get("CPP_MEMORY_DECAY", "0.92"))
        self.memory_trace_gain = float(os.environ.get("CPP_MEMORY_TRACE_GAIN", "0.35"))
        self.relative_memory_horizon = float(os.environ.get("CPP_RELATIVE_MEMORY_HORIZON", "12"))
        self.direction_traces = np.zeros(4, dtype=np.float32)
        self.action_memory_decay = float(os.environ.get("CPP_ACTION_MEMORY_DECAY", "0.90"))
        self.action_new_cell_ema = np.zeros(4, dtype=np.float32)
        self.action_stale_ema = np.zeros(4, dtype=np.float32)
        self.state_variant = os.environ.get("CPP_STATE_VARIANT", "local_history_v2")
        self.reward_variant = os.environ.get("CPP_REWARD_VARIANT", "original")
        self.local_view_size = int(os.environ.get("CPP_LOCAL_VIEW_SIZE", "3"))
        if self.local_view_size % 2 == 0 or self.local_view_size < 3:
            raise ValueError("CPP_LOCAL_VIEW_SIZE must be an odd integer >= 3")
        self.local_view_center = self.local_view_size // 2

        # Track visited cells
        self.visited = set()

        self._agent_location = np.array([-1, -1], dtype=int)
        self._neighbors = np.zeros((self.local_view_size, self.local_view_size), dtype=int)

        # Observation: Dict with agent info (x, y, coverage) and 3x3 neighbor matrix
        if self.state_variant == "local_direction_v1":
            agent_obs_size = 36
        elif self.state_variant == "local_relative_trace_v1":
            agent_obs_size = 30
        elif self.state_variant == "local_memory_decay_v3":
            agent_obs_size = 52
        elif self.state_variant == "local_memory_decay_v2":
            agent_obs_size = 36
        elif self.state_variant == "local_memory_decay_v1":
            agent_obs_size = 30
        elif self.state_variant == "local_memory_v1":
            agent_obs_size = 30
        elif self.state_variant == "local_history_v4":
            agent_obs_size = 40
        elif self.state_variant == "local_history_v3":
            agent_obs_size = 32
        elif self.state_variant == "local_history_v2":
            agent_obs_size = 24
        else:
            agent_obs_size = 12
        self.observation_space = gym.spaces.Dict({
            "agent": gym.spaces.Box(
                low=np.zeros(agent_obs_size, dtype=np.float32),
                high=np.ones(agent_obs_size, dtype=np.float32),
                dtype=np.float32
            ),
            "neighbors": gym.spaces.Box(
                low=np.zeros((self.local_view_size, self.local_view_size), dtype=np.float32),
                high=np.full((self.local_view_size, self.local_view_size), 2.0, dtype=np.float32),
                dtype=np.float32
            ),
        })

        # 4 actions: right, up, left, down
        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([1, 0]),   # right
            1: np.array([0, -1]),  # up
            2: np.array([-1, 0]),  # left
            3: np.array([0, 1]),   # down
        }

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

    @property
    def total_free_cells(self):
        return self.size * self.size - len(self.obstacles_locations)

    @property
    def coverage_ratio(self):
        return len(self.visited) / self.total_free_cells if self.total_free_cells > 0 else 1.0

    def _all_free_cells_reachable(self):
        """
        Checks whether all non-obstacle cells are reachable from the agent's
        initial position. This is used only by the environment during reset,
        and is not exposed to the agent.
        """
        obstacles = {tuple(loc) for loc in self.obstacles_locations}
        start = tuple(self._agent_location)

        visited = set()
        stack = [start]

        while stack:
            cell = stack.pop()

            if cell in visited:
                continue

            visited.add(cell)
            x, y = cell

            for direction in self._action_to_direction.values():
                nx = x + int(direction[0])
                ny = y + int(direction[1])
                neighbor = (nx, ny)

                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    continue

                if neighbor in obstacles:
                    continue

                if neighbor not in visited:
                    stack.append(neighbor)

        total_free_cells = self.size * self.size - len(obstacles)

        return len(visited) == total_free_cells

    def _get_obs(self):
        # Local statistics computed only from the partial observation window.
        # These values do not reveal the full map.
        local_cell_count = float(self.local_view_size * self.local_view_size)
        c = self.local_view_center
        local_unvisited_ratio = np.sum(self._neighbors == 0) / local_cell_count
        local_blocked_ratio = np.sum(self._neighbors == 1) / local_cell_count
        local_visited_ratio = np.sum(self._neighbors == 2) / local_cell_count
        adjacent_cells = np.array([
            self._neighbors[c, c + 1],
            self._neighbors[c - 1, c],
            self._neighbors[c, c - 1],
            self._neighbors[c + 1, c],
        ])
        adjacent_unvisited_ratio = np.sum(adjacent_cells == 0) / 4.0
        has_local_unvisited = float(np.any(self._neighbors == 0))
        step_ratio = min(self.count_steps / self.max_steps, 1.0)
        steps_since_new_cell_ratio = min(self.steps_since_new_cell / self.max_steps, 1.0)
        stuck_short_ratio = min(self.steps_since_new_cell / 50.0, 1.0)

        if self.state_variant in {"local_history_v2", "local_history_v3", "local_history_v4", "local_direction_v1", "local_memory_v1", "local_memory_decay_v1", "local_memory_decay_v2", "local_memory_decay_v3", "local_relative_trace_v1"}:
            adjacent_blocked_ratio = np.sum(adjacent_cells == 1) / 4.0
            last_action_one_hot = np.zeros(4, dtype=np.float32)
            previous_action_one_hot = np.zeros(4, dtype=np.float32)
            last_new_action_one_hot = np.zeros(4, dtype=np.float32)

            if 0 <= self.last_action < 4:
                last_action_one_hot[self.last_action] = 1.0
            if 0 <= self.previous_action < 4:
                previous_action_one_hot[self.previous_action] = 1.0
            if 0 <= self.last_new_action < 4:
                last_new_action_one_hot[self.last_new_action] = 1.0

            reversed_last_action = float(
                self.last_action >= 0
                and self.previous_action >= 0
                and self.last_action == (self.previous_action + 2) % 4
            )
            repeated_last_action = float(
                self.last_action >= 0
                and self.previous_action >= 0
                and self.last_action == self.previous_action
            )
            recent_new_cell_ratio = (
                sum(self.recent_new_cells) / len(self.recent_new_cells)
                if self.recent_new_cells else 0.0
            )

            agent_features = [
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
                local_unvisited_ratio,
                local_blocked_ratio,
                local_visited_ratio,
                adjacent_unvisited_ratio,
                adjacent_blocked_ratio,
                has_local_unvisited,
                self.last_collision,
                self.last_new_cell,
                stuck_short_ratio,
                step_ratio,
                reversed_last_action,
                repeated_last_action,
                recent_new_cell_ratio,
                *last_action_one_hot,
                *previous_action_one_hot,
            ]

            if self.state_variant == "local_direction_v1":
                action_cell_values = [
                    self._neighbors[c, c + 1],
                    self._neighbors[c - 1, c],
                    self._neighbors[c, c - 1],
                    self._neighbors[c + 1, c],
                ]
                action_unvisited = [float(value == 0) for value in action_cell_values]
                action_blocked = [float(value == 1) for value in action_cell_values]
                action_visited = [float(value == 2) for value in action_cell_values]
                agent_features.extend([
                    *action_unvisited,
                    *action_blocked,
                    *action_visited,
                ])

            if self.state_variant == "local_history_v3":
                max_index = max(self.size - 1, 1)
                current_pos = tuple(self._agent_location)
                current_cell_revisit_ratio = min(self.current_cell_visits.get(current_pos, 0) / 10.0, 1.0)
                recent_collision_ratio = (
                    sum(self.recent_collisions) / len(self.recent_collisions)
                    if self.recent_collisions else 0.0
                )
                agent_features.extend([
                    self._agent_location[0] / max_index,
                    (max_index - self._agent_location[0]) / max_index,
                    self._agent_location[1] / max_index,
                    (max_index - self._agent_location[1]) / max_index,
                    float(self._agent_location[1] % 2),
                    float(self._agent_location[0] % 2),
                    current_cell_revisit_ratio,
                    recent_collision_ratio,
                ])

            if self.state_variant == "local_history_v4":
                current_pos = tuple(self._agent_location)
                current_cell_revisit_ratio = min(self.current_cell_visits.get(current_pos, 0) / 10.0, 1.0)
                recent_collision_ratio = (
                    sum(self.recent_collisions) / len(self.recent_collisions)
                    if self.recent_collisions else 0.0
                )
                stuck_medium_ratio = min(self.steps_since_new_cell / 100.0, 1.0)
                stuck_long_ratio = min(self.steps_since_new_cell / 200.0, 1.0)
                recent_action_distribution = np.zeros(4, dtype=np.float32)
                recent_new_action_distribution = np.zeros(4, dtype=np.float32)

                if self.recent_actions:
                    for recent_action in self.recent_actions:
                        recent_action_distribution[recent_action] += 1.0
                    recent_action_distribution /= len(self.recent_actions)

                recent_new_count = sum(self.recent_action_new_cells)
                if recent_new_count > 0:
                    for recent_action, new_cell in zip(self.recent_actions, self.recent_action_new_cells):
                        if new_cell:
                            recent_new_action_distribution[recent_action] += 1.0
                    recent_new_action_distribution /= recent_new_count

                agent_features.extend([
                    current_cell_revisit_ratio,
                    recent_collision_ratio,
                    stuck_medium_ratio,
                    stuck_long_ratio,
                    *last_new_action_one_hot,
                    *recent_action_distribution,
                    *recent_new_action_distribution,
                ])

            if self.state_variant == "local_memory_v1":
                current_pos = tuple(self._agent_location)
                self.observed_unvisited_cells.difference_update(self.visited)
                has_memory_target = float(len(self.observed_unvisited_cells) > 0)
                nearest_dx = 0.0
                nearest_dy = 0.0
                nearest_distance_ratio = 0.0

                if self.observed_unvisited_cells:
                    nearest_cell = min(
                        self.observed_unvisited_cells,
                        key=lambda cell: abs(cell[0] - current_pos[0]) + abs(cell[1] - current_pos[1]),
                    )
                    nearest_distance = abs(nearest_cell[0] - current_pos[0]) + abs(nearest_cell[1] - current_pos[1])
                    max_axis = max(self.size - 1, 1)
                    max_distance = max(2 * (self.size - 1), 1)
                    nearest_dx = ((nearest_cell[0] - current_pos[0]) / max_axis + 1.0) / 2.0
                    nearest_dy = ((nearest_cell[1] - current_pos[1]) / max_axis + 1.0) / 2.0
                    nearest_distance_ratio = nearest_distance / max_distance

                memory_count_ratio = min(len(self.observed_unvisited_cells) / max(self.total_free_cells, 1), 1.0)
                late_game_ratio = min(max(self.coverage_ratio - 0.85, 0.0) / 0.15, 1.0)
                agent_features.extend([
                    has_memory_target,
                    nearest_dx,
                    nearest_dy,
                    nearest_distance_ratio,
                    memory_count_ratio,
                    late_game_ratio,
                ])

            if self.state_variant == "local_memory_decay_v1":
                memory_dx_normalized = (np.clip(self.memory_dx, -1.0, 1.0) + 1.0) / 2.0
                memory_dy_normalized = (np.clip(self.memory_dy, -1.0, 1.0) + 1.0) / 2.0
                memory_age_ratio = min(self.memory_age / 100.0, 1.0)
                late_game_ratio = min(max(self.coverage_ratio - 0.85, 0.0) / 0.15, 1.0)
                agent_features.extend([
                    memory_dx_normalized,
                    memory_dy_normalized,
                    min(self.memory_strength, 1.0),
                    memory_age_ratio,
                    self.last_observed_opportunity,
                    late_game_ratio,
                ])

            if self.state_variant == "local_relative_trace_v1":
                horizon = max(self.relative_memory_horizon, 1.0)
                memory_dx_normalized = (np.clip(self.memory_dx / horizon, -1.0, 1.0) + 1.0) / 2.0
                memory_dy_normalized = (np.clip(self.memory_dy / horizon, -1.0, 1.0) + 1.0) / 2.0
                memory_distance_ratio = min((abs(self.memory_dx) + abs(self.memory_dy)) / horizon, 1.0)
                memory_age_ratio = min(self.memory_age / 40.0, 1.0)
                late_game_ratio = min(max(self.coverage_ratio - 0.85, 0.0) / 0.15, 1.0)
                agent_features.extend([
                    memory_dx_normalized,
                    memory_dy_normalized,
                    memory_distance_ratio,
                    min(self.memory_strength, 1.0),
                    memory_age_ratio,
                    late_game_ratio,
                ])

            if self.state_variant in {"local_memory_decay_v2", "local_memory_decay_v3"}:
                traces = np.clip(self.direction_traces, 0.0, 1.0)
                best_trace_one_hot = np.zeros(4, dtype=np.float32)
                max_trace = float(np.max(traces))

                if max_trace > 0.05:
                    best_trace_one_hot[int(np.argmax(traces))] = 1.0

                memory_dx_normalized = (np.clip(self.memory_dx, -1.0, 1.0) + 1.0) / 2.0
                memory_dy_normalized = (np.clip(self.memory_dy, -1.0, 1.0) + 1.0) / 2.0
                late_game_ratio = min(max(self.coverage_ratio - 0.85, 0.0) / 0.15, 1.0)
                agent_features.extend([
                    *traces,
                    *best_trace_one_hot,
                    memory_dx_normalized,
                    memory_dy_normalized,
                    max_trace,
                    late_game_ratio,
                ])

                if self.state_variant == "local_memory_decay_v3":
                    current_pos = tuple(self._agent_location)
                    current_cell_revisit_ratio = min(self.current_cell_visits.get(current_pos, 0) / 10.0, 1.0)
                    recent_collision_ratio = (
                        sum(self.recent_collisions) / len(self.recent_collisions)
                        if self.recent_collisions else 0.0
                    )
                    stuck_medium_ratio = min(self.steps_since_new_cell / 100.0, 1.0)
                    stuck_long_ratio = min(self.steps_since_new_cell / 200.0, 1.0)
                    agent_features.extend([
                        *np.clip(self.action_new_cell_ema, 0.0, 1.0),
                        *np.clip(self.action_stale_ema, 0.0, 1.0),
                        *last_new_action_one_hot,
                        current_cell_revisit_ratio,
                        recent_collision_ratio,
                        stuck_medium_ratio,
                        stuck_long_ratio,
                    ])

            return {
                "agent": np.array(agent_features, dtype=np.float32),
                "neighbors": self._neighbors.astype(np.float32),
            }

        return {
            "agent": np.array([
                self._agent_location[0] / self.size,
                self._agent_location[1] / self.size,
                self.coverage_ratio,
                local_unvisited_ratio,
                local_blocked_ratio,
                local_visited_ratio,
                self.last_action_normalized,
                self.last_collision,
                self.last_new_cell,
                steps_since_new_cell_ratio,
                step_ratio,
                adjacent_unvisited_ratio * has_local_unvisited,
            ], dtype=np.float32),
            "neighbors": self._neighbors.astype(np.float32),
        }

    def _get_info(self):
        return {
            "coverage": self.coverage_ratio,
            "visited_cells": len(self.visited),
            "total_free_cells": self.total_free_cells,
            "steps": self.count_steps,
            "size": self.size,
        }

    def _update_directional_memory(self):
        if self.state_variant not in {"local_memory_decay_v1", "local_memory_decay_v2", "local_memory_decay_v3", "local_relative_trace_v1"}:
            return

        c = self.local_view_center
        offsets = []

        for i in range(self.local_view_size):
            for j in range(self.local_view_size):
                if self._neighbors[i, j] == 0 and (i != c or j != c):
                    offsets.append((j - c, i - c))

        decay = np.clip(self.memory_decay, 0.0, 1.0)

        if self.state_variant == "local_relative_trace_v1":
            horizon = max(self.relative_memory_horizon, 1.0)

            if not self.last_collision and 0 <= self.last_action < 4:
                movement = self._action_to_direction[self.last_action]
                self.memory_dx -= float(movement[0])
                self.memory_dy -= float(movement[1])

            self.memory_dx = float(np.clip(self.memory_dx, -horizon, horizon))
            self.memory_dy = float(np.clip(self.memory_dy, -horizon, horizon))

            if offsets:
                if self.memory_strength > 0.05:
                    offsets_array = np.array(offsets, dtype=np.float32)
                    memory_vector = np.array([self.memory_dx, self.memory_dy], dtype=np.float32)
                    alignment = offsets_array @ memory_vector
                    distances = np.abs(offsets_array[:, 0]) + np.abs(offsets_array[:, 1])
                    selected = offsets_array[int(np.argmax(alignment - 0.25 * distances))]
                    observed_dx = float(selected[0])
                    observed_dy = float(selected[1])
                else:
                    observed_dx, observed_dy = min(offsets, key=lambda item: abs(item[0]) + abs(item[1]))

                self.memory_dx = 0.35 * self.memory_dx + 0.65 * float(observed_dx)
                self.memory_dy = 0.35 * self.memory_dy + 0.65 * float(observed_dy)
                self.memory_strength = min(1.0, decay * self.memory_strength + 0.45)
                self.memory_age = 0.0
                self.last_observed_opportunity = 1.0
            else:
                self.memory_dx *= decay
                self.memory_dy *= decay
                self.memory_strength *= decay
                self.memory_age += 1.0
                self.last_observed_opportunity = 0.0

            if self.last_new_cell and (abs(self.memory_dx) + abs(self.memory_dy)) <= 1.5:
                self.memory_strength *= 0.35
            if self.memory_age > 40.0 or self.memory_strength < 0.02:
                self.memory_dx = 0.0
                self.memory_dy = 0.0
                self.memory_strength = 0.0
            return

        if self.state_variant in {"local_memory_decay_v2", "local_memory_decay_v3"}:
            direction_scores = np.zeros(4, dtype=np.float32)

            for dx, dy in offsets:
                distance = max(abs(dx) + abs(dy), 1)
                if dx > 0:
                    direction_scores[0] += abs(dx) / distance
                elif dx < 0:
                    direction_scores[2] += abs(dx) / distance
                if dy < 0:
                    direction_scores[1] += abs(dy) / distance
                elif dy > 0:
                    direction_scores[3] += abs(dy) / distance

            direction_scores = np.clip(direction_scores, 0.0, 1.0)
            gain = max(self.memory_trace_gain, 0.0)
            self.direction_traces = np.clip(
                decay * self.direction_traces + gain * direction_scores,
                0.0,
                1.0,
            )

            if offsets:
                self.memory_age = 0.0
                self.last_observed_opportunity = 1.0
            else:
                self.memory_age += 1.0
                self.last_observed_opportunity = 0.0

            if self.last_collision and 0 <= self.last_action < 4:
                self.direction_traces[self.last_action] *= 0.2
            elif self.last_new_cell and 0 <= self.last_action < 4:
                self.direction_traces[self.last_action] *= 0.35

            self.memory_dx = float(self.direction_traces[0] - self.direction_traces[2])
            self.memory_dy = float(self.direction_traces[3] - self.direction_traces[1])
            self.memory_strength = float(np.max(self.direction_traces))
            return

        if offsets:
            offsets_array = np.array(offsets, dtype=np.float32)
            normalizer = max(float(c), 1.0)
            observed_dx = float(np.mean(offsets_array[:, 0]) / normalizer)
            observed_dy = float(np.mean(offsets_array[:, 1]) / normalizer)
            observed_strength = min(len(offsets) / max(self.local_view_size * self.local_view_size - 1, 1), 1.0)

            self.memory_dx = decay * self.memory_dx + (1.0 - decay) * observed_dx
            self.memory_dy = decay * self.memory_dy + (1.0 - decay) * observed_dy
            self.memory_strength = decay * self.memory_strength + (1.0 - decay) * observed_strength
            self.memory_age = 0.0
            self.last_observed_opportunity = 1.0
        else:
            self.memory_dx *= decay
            self.memory_dy *= decay
            self.memory_strength *= decay
            self.memory_age += 1.0
            self.last_observed_opportunity = 0.0

        if self.last_new_cell:
            self.memory_dx *= 0.8
            self.memory_dy *= 0.8
            self.memory_strength *= 0.5

    def _update_action_memory(self):
        if self.state_variant != "local_memory_decay_v3" or not (0 <= self.last_action < 4):
            return

        decay = np.clip(self.action_memory_decay, 0.0, 1.0)
        action = self.last_action
        self.action_new_cell_ema *= decay
        self.action_stale_ema *= decay

        if self.last_new_cell:
            self.action_new_cell_ema[action] = min(1.0, self.action_new_cell_ema[action] + 0.5)
            self.action_stale_ema[action] *= 0.25
        elif self.last_collision:
            self.action_new_cell_ema[action] *= 0.25
            self.action_stale_ema[action] = min(1.0, self.action_stale_ema[action] + 0.7)
        else:
            self.action_stale_ema[action] = min(1.0, self.action_stale_ema[action] + 0.25)

    def set_neighbors(self, obstacles_locations):
        # Create a local matrix centered on the agent's location.
        # Row index i corresponds to agent_y + (i-center), col index j to agent_x + (j-center).
        # 0 = free (not yet visited), 1 = obstacle or wall (out-of-bounds), 2 = already visited.
        matrix = np.zeros((self.local_view_size, self.local_view_size), dtype=int)
        c = self.local_view_center
        for i in range(self.local_view_size):
            for j in range(self.local_view_size):
                nx = self._agent_location[0] + (j - c)
                ny = self._agent_location[1] + (i - c)
                if not (0 <= nx < self.size and 0 <= ny < self.size):
                    matrix[i][j] = 1
                elif (nx, ny) in obstacles_locations:
                    matrix[i][j] = 1
                elif (nx, ny) in self.visited:
                    matrix[i][j] = 2
                elif self.state_variant == "local_memory_v1":
                    self.observed_unvisited_cells.add((nx, ny))
        if self.state_variant == "local_memory_v1":
            self.observed_unvisited_cells.difference_update(self.visited)
        self._neighbors = matrix

    def action_masks(self):
        """
        Valid action mask derived only from immediate local movement validity.
        This prevents actions that would hit a wall or adjacent obstacle,
        without exposing any global map structure to the policy.
        """
        masks = []
        obstacles = {tuple(loc) for loc in self.obstacles_locations}

        for action in range(self.action_space.n):
            direction = self._action_to_direction[action]
            target = self._agent_location + direction
            target_tuple = tuple(target)

            valid = (
                0 <= target[0] < self.size
                and 0 <= target[1] < self.size
                and target_tuple not in obstacles
            )
            masks.append(valid)

        return np.array(masks, dtype=bool)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.count_steps = 0
        self.obstacles_locations = []
        self.visited = set()
        self.last_action_normalized = 0.0
        self.last_action = -1
        self.previous_action = -1
        self.last_new_action = -1
        self.last_collision = 0.0
        self.last_new_cell = 0.0
        self.steps_since_new_cell = 0
        self.recent_new_cells = []
        self.recent_collisions = []
        self.recent_actions = []
        self.recent_action_new_cells = []
        self.current_cell_visits = {}
        self.observed_unvisited_cells = set()
        self.memory_dx = 0.0
        self.memory_dy = 0.0
        self.memory_strength = 0.0
        self.memory_age = 1.0
        self.last_observed_opportunity = 0.0
        self.direction_traces = np.zeros(4, dtype=np.float32)
        self.action_new_cell_ema = np.zeros(4, dtype=np.float32)
        self.action_stale_ema = np.zeros(4, dtype=np.float32)

        # Place agent randomly
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        # Place obstacles.
        # We resample obstacles until all free cells are reachable from the
        # agent's initial position. This avoids impossible CPP episodes.
        valid_map = False

        while not valid_map:
            self.obstacles_locations = []

            for _ in range(self.obs_quantity):
                obstacle_location = self._agent_location.copy()

                while (
                    np.array_equal(obstacle_location, self._agent_location)
                    or tuple(obstacle_location) in self.obstacles_locations
                ):
                    obstacle_location = self.np_random.integers(0, self.size, size=2, dtype=int)

                self.obstacles_locations.append(tuple(obstacle_location))

            valid_map = self._all_free_cells_reachable()

        # Mark starting position as visited
        self.visited.add(tuple(self._agent_location))
        self.current_cell_visits[tuple(self._agent_location)] = 1

        self.set_neighbors(self.obstacles_locations)
        self._update_directional_memory()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, info

    def step(self, action):

        direction = self._action_to_direction[action]
        old_location = self._agent_location.copy()

        # Move agent (clip to grid bounds)
        self._agent_location = np.clip(
            self._agent_location + direction, 0, self.size - 1
        )

        # If the agent hits an obstacle, stay in place
        if tuple(self._agent_location) in self.obstacles_locations:
            self._agent_location = old_location

        self.set_neighbors(self.obstacles_locations)
        self.count_steps += 1

        # --- CPP Reward Function ---
        current_pos = tuple(self._agent_location)
        is_new_cell = current_pos not in self.visited
        stayed_in_place = np.array_equal(self._agent_location, old_location)
        self.previous_action = self.last_action
        self.last_action = int(action)
        self.last_action_normalized = action / 3.0
        self.last_collision = float(stayed_in_place)
        self.last_new_cell = float(is_new_cell and not stayed_in_place)
        self.recent_new_cells.append(self.last_new_cell)
        self.recent_new_cells = self.recent_new_cells[-20:]
        self.recent_collisions.append(self.last_collision)
        self.recent_collisions = self.recent_collisions[-20:]
        self.recent_actions.append(int(action))
        self.recent_actions = self.recent_actions[-20:]
        self.recent_action_new_cells.append(self.last_new_cell)
        self.recent_action_new_cells = self.recent_action_new_cells[-20:]
        self.current_cell_visits[current_pos] = self.current_cell_visits.get(current_pos, 0) + 1

        if self.last_new_cell:
            self.steps_since_new_cell = 0
            self.last_new_action = int(action)
        else:
            self.steps_since_new_cell += 1

        self._update_action_memory()
        self._update_directional_memory()

        if self.reward_variant == "finish_aware_v1":
            progress = self.coverage_ratio
            reward = -0.05

            if stayed_in_place:
                reward -= 0.5
            elif is_new_cell:
                reward += 1.0 + progress
                self.visited.add(current_pos)
            else:
                reward -= 0.2 + 0.6 * progress
        elif self.reward_variant == "large_grid_v1":
            # Large grids require long transfers through already visited cells.
            # This shaping uses only current transition facts and local context.
            reward = -0.03
            c = self.local_view_center
            has_adjacent_unvisited = np.any(np.array([
                self._neighbors[c, c + 1],
                self._neighbors[c - 1, c],
                self._neighbors[c, c - 1],
                self._neighbors[c + 1, c],
            ]) == 0)

            if stayed_in_place:
                reward -= 0.5
            elif is_new_cell:
                reward += 1.0 + 0.5 * self.coverage_ratio
                self.visited.add(current_pos)
            elif has_adjacent_unvisited:
                reward -= 0.25
            else:
                reward -= 0.05
        elif self.reward_variant == "late_finish_v1":
            progress = self.coverage_ratio
            reward = -0.08

            if stayed_in_place:
                reward -= 0.6
            elif is_new_cell:
                late_bonus = max(0.0, progress - 0.85) / 0.15
                reward += 1.0 + progress + min(1.0, late_bonus)
                self.visited.add(current_pos)
            else:
                late_revisit_penalty = 0.35 * max(0.0, progress - 0.85) / 0.15
                reward -= 0.2 + min(0.35, late_revisit_penalty)

                if self.steps_since_new_cell > 35:
                    reward -= min(0.15, 0.005 * (self.steps_since_new_cell - 35))
        else:
            # Base step penalty
            reward = -0.1

            if stayed_in_place:
                # Hitting wall or obstacle.
                reward -= 0.5
            elif is_new_cell:
                # Reward for exploring new cell.
                reward += 1.0
                self.visited.add(current_pos)
            else:
                # Penalty for revisiting an already visited cell.
                reward -= 0.3

            if self.reward_variant == "anti_stagnation_v1" and self.steps_since_new_cell > 30:
                reward -= min(0.2, 0.01 * (self.steps_since_new_cell - 30))

        # Check if full coverage achieved
        full_coverage = len(self.visited) >= self.total_free_cells
        terminated = full_coverage

        if full_coverage:
            if self.reward_variant == "large_grid_v1":
                reward += 25.0
            elif self.reward_variant == "late_finish_v1":
                reward += 35.0
            elif self.reward_variant == "finish_aware_v1":
                reward += 20.0
            elif self.reward_variant == "finish_bonus_30":
                reward += 30.0
            else:
                reward += 10.0

        # Truncation on max steps
        if self.count_steps >= self.max_steps and not terminated:
            truncated = True
            if self.reward_variant == "large_grid_v1":
                missing_ratio = 1.0 - self.coverage_ratio
                reward -= 5.0 + 20.0 * missing_ratio
            elif self.reward_variant == "late_finish_v1":
                missing_ratio = 1.0 - self.coverage_ratio
                reward -= 5.0 + 20.0 * missing_ratio
            elif self.reward_variant == "finish_aware_v1":
                reward -= 10.0
            else:
                reward -= 5.0
        else:
            truncated = False

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self._render_frame()

        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size

        # Draw visited cells in light green
        for cell in self.visited:
            cell_arr = np.array(cell)
            pygame.draw.rect(
                canvas,
                (144, 238, 144),  # light green
                pygame.Rect(
                    pix_square_size * cell_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw obstacles in black
        for obs in self.obstacles_locations:
            obs_arr = np.array(obs)
            pygame.draw.rect(
                canvas,
                (0, 0, 0),
                pygame.Rect(
                    pix_square_size * obs_arr,
                    (pix_square_size, pix_square_size),
                ),
            )

        # Draw agent as blue circle
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            (self._agent_location + 0.5) * pix_square_size,
            pix_square_size / 3,
        )

        # Draw coverage info text
        font = pygame.font.SysFont(None, 24)
        coverage_text = font.render(
            f"Coverage: {self.coverage_ratio:.1%} | Steps: {self.count_steps}",
            True, (0, 0, 0)
        )
        canvas.blit(coverage_text, (5, 5))

        # Draw gridlines
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x),
                             (self.window_size, pix_square_size * x), width=3)
            pygame.draw.line(canvas, 0, (pix_square_size * x, 0),
                             (pix_square_size * x, self.window_size), width=3)

        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
