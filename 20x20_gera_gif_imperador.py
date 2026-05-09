from pathlib import Path
import importlib.util
import os

os.environ["CPP_STATE_VARIANT"] = "local_memory_v1"
os.environ["CPP_REWARD_VARIANT"] = "original"
os.environ["CPP_LOCAL_VIEW_SIZE"] = "3"

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image
from stable_baselines3 import PPO


MODEL_PATH = Path("data/20x20_experiments/Imperador_20X20.zip")
OUTPUT_PATH = Path("Graficos/15_imperador_20x20.gif")


def load_env():
    env_path = Path(__file__).with_name("20x20_grid_world_cpp.py")
    spec = importlib.util.spec_from_file_location("grid_world_cpp_20x20_experiment", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load environment from {env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GridWorldCPPEnv


def register_env(env_class):
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP20x20-v0",
            entry_point=env_class,
        )
    except Exception:
        pass


def sample_frames(frames, max_frames=140):
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]


def save_gif(frames):
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    images = [
        Image.fromarray(frame).resize((384, 384), Image.Resampling.NEAREST)
        for frame in sample_frames(frames)
    ]
    images.extend([images[-1]] * 8)
    images[0].save(
        OUTPUT_PATH,
        save_all=True,
        append_images=images[1:],
        duration=85,
        loop=0,
        optimize=True,
    )


def run_episode(model, seed):
    env = gym.make(
        "gymnasium_env/GridWorldCPP20x20-v0",
        size=20,
        obs_quantity=48,
        max_steps=1000,
        render_mode="rgb_array",
    )
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    done = False
    truncated = False
    steps = 0
    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, done, truncated, info = env.step(action.item())
        frames.append(env.render())
        steps += 1
    env.close()
    return done and not truncated, float(info["coverage"]), steps, frames


def main():
    pygame.init()
    pygame.font.init()
    register_env(load_env())
    model = PPO.load(str(MODEL_PATH), device="cpu")

    best = None
    for seed in range(30000, 30100):
        success, coverage, steps, frames = run_episode(model, seed)
        result = (success, coverage, -steps, seed, steps, frames)
        if best is None or result[:3] > best[:3]:
            best = result
        if success and steps < 1000:
            break

    success, coverage, _, seed, steps, frames = best
    save_gif(frames)
    print(f"Imperador 20x20: success={success} coverage={coverage:.2%} steps={steps} seed={seed}")
    pygame.quit()


if __name__ == "__main__":
    main()
