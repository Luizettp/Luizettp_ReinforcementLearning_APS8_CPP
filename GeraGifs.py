from pathlib import Path
import importlib.util
import os

import gymnasium as gym
import numpy as np
import pygame
from PIL import Image
from stable_baselines3 import PPO

os.environ.setdefault("CPP_STATE_VARIANT", "local_history_v2")
os.environ.setdefault("CPP_REWARD_VARIANT", "original")
os.environ.setdefault("CPP_LOCAL_VIEW_SIZE", "3")

MODEL_PATH = Path("data/20x20_experiments/Gladiador_2020.zip")
OUTPUT_DIR = Path("Graficos")
OUTPUT_DIR.mkdir(exist_ok=True)


def load_experimental_env():
    env_path = Path(__file__).with_name("20x20_grid_world_cpp.py")
    spec = importlib.util.spec_from_file_location("grid_world_cpp_20x20_experiment", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experimental environment from {env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GridWorldCPPEnv


GridWorldCPPEnv = load_experimental_env()


def register_env():
    try:
        gym.register(
            id="gymnasium_env/GridWorldCPP20x20-v0",
            entry_point=GridWorldCPPEnv,
        )
    except Exception:
        pass


def resize_frame(frame, size=384):
    return Image.fromarray(frame).resize((size, size), Image.Resampling.NEAREST)


def sample_frames(frames, max_frames=140):
    if len(frames) <= max_frames:
        return frames

    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]


def run_episode(model, dim, obstacles, max_steps, seed, deterministic):
    env = gym.make(
        "gymnasium_env/GridWorldCPP20x20-v0",
        size=dim,
        obs_quantity=obstacles,
        max_steps=max_steps,
        render_mode="rgb_array",
    )

    obs, info = env.reset(seed=seed)
    frames = [env.render()]

    done = False
    truncated = False
    steps = 0

    while not done and not truncated:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action.item())
        frames.append(env.render())
        steps += 1

    env.close()

    return {
        "success": bool(done and not truncated),
        "coverage": float(info["coverage"]),
        "steps": steps,
        "frames": frames,
        "seed": seed,
        "deterministic": deterministic,
    }


def find_good_episode(model, dim, obstacles, max_steps, base_seed, attempts):
    candidates = []

    for attempt in range(attempts):
        seed = base_seed + attempt
        result = run_episode(
            model=model,
            dim=dim,
            obstacles=obstacles,
            max_steps=max_steps,
            seed=seed,
            deterministic=False,
        )

        candidates.append(result)

        if result["success"]:
            return result

    return max(candidates, key=lambda item: (item["coverage"], -item["steps"]))


def save_gif(result, output_path):
    frames = [resize_frame(frame) for frame in sample_frames(result["frames"])]

    # Repeat the final frame so the completed coverage is visible for a moment.
    frames.extend([frames[-1]] * 8)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=85,
        loop=0,
        optimize=True,
    )


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    pygame.init()
    pygame.font.init()
    register_env()

    model = PPO.load(str(MODEL_PATH), device="cpu")

    jobs = [
        {
            "label": "5x5",
            "dim": 5,
            "obstacles": 3,
            "max_steps": 200,
            "seed": 5000,
            "attempts": 80,
            "output": OUTPUT_DIR / "09_visualizacao_modelo_final_5x5.gif",
        },
        {
            "label": "10x10",
            "dim": 10,
            "obstacles": 12,
            "max_steps": 500,
            "seed": 10000,
            "attempts": 200,
            "output": OUTPUT_DIR / "10_visualizacao_modelo_final_10x10.gif",
        },
        {
            "label": "20x20",
            "dim": 20,
            "obstacles": 48,
            "max_steps": 1000,
            "seed": 20000,
            "attempts": 500,
            "output": OUTPUT_DIR / "11_gladiador_2020_20x20.gif",
        },
    ]

    for job in jobs:
        result = find_good_episode(
            model=model,
            dim=job["dim"],
            obstacles=job["obstacles"],
            max_steps=job["max_steps"],
            base_seed=job["seed"],
            attempts=job["attempts"],
        )
        save_gif(result, job["output"])
        print(
            f"{job['label']}: saved {job['output']} | "
            f"success={result['success']} coverage={result['coverage']:.2%} "
            f"steps={result['steps']} seed={result['seed']} "
            f"deterministic={result['deterministic']}"
        )

    pygame.quit()


if __name__ == "__main__":
    main()
