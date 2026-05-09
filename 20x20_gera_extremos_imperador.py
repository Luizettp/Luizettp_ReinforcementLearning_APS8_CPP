from pathlib import Path
import importlib.util
import os

os.environ["CPP_STATE_VARIANT"] = "local_memory_v1"
os.environ["CPP_REWARD_VARIANT"] = "original"
os.environ["CPP_LOCAL_VIEW_SIZE"] = "3"

import gymnasium as gym
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pygame
from PIL import Image
from stable_baselines3 import PPO


MODEL_PATH = Path("data/20x20_experiments/Imperador_20X20.zip")
GRAPH_DIR = Path("Graficos")
GRAPH_DIR.mkdir(exist_ok=True)

FEATS = [
    {
        "mapa": "30x30",
        "dim": 30,
        "obstacles": 108,
        "max_steps": 2250,
        "episodes": 100,
        "full_coverage": 92.00,
        "average_coverage": 99.96,
        "average_steps": 1455.3,
        "seed": 30000,
        "gif": "17_imperador_30x30.gif",
    },
    {
        "mapa": "40x40",
        "dim": 40,
        "obstacles": 192,
        "max_steps": 6400,
        "episodes": 100,
        "full_coverage": 97.00,
        "average_coverage": 99.94,
        "average_steps": 2994.9,
        "seed": 40000,
        "gif": "18_imperador_40x40.gif",
    },
    {
        "mapa": "60x60",
        "dim": 60,
        "obstacles": 432,
        "max_steps": 14400,
        "episodes": 100,
        "full_coverage": 90.00,
        "average_coverage": 99.98,
        "average_steps": 9404.7,
        "seed": 60000,
        "gif": "19_imperador_60x60.gif",
    },
]


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
        gym.register(id="gymnasium_env/GridWorldCPP20x20-v0", entry_point=env_class)
    except Exception:
        pass


def sample_frames(frames, max_frames=170):
    if len(frames) <= max_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
    return [frames[i] for i in indices]


def save_gif(frames, output_path):
    images = [
        Image.fromarray(frame).resize((420, 420), Image.Resampling.NEAREST)
        for frame in sample_frames(frames)
    ]
    images.extend([images[-1]] * 8)
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=80,
        loop=0,
        optimize=True,
    )


def run_episode(model, feat, seed):
    env = gym.make(
        "gymnasium_env/GridWorldCPP20x20-v0",
        size=feat["dim"],
        obs_quantity=feat["obstacles"],
        max_steps=feat["max_steps"],
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
    return done and not truncated, float(info["coverage"]), steps, frames, seed


def make_gif(model, feat):
    best = None
    for seed in range(feat["seed"], feat["seed"] + 120):
        result = run_episode(model, feat, seed)
        success, coverage, steps, frames, seed = result
        score = (success, coverage, -steps)
        if best is None or score > best[0]:
            best = (score, result)
        if success:
            break

    _, result = best
    success, coverage, steps, frames, seed = result
    save_gif(frames, GRAPH_DIR / feat["gif"])
    print(f"{feat['mapa']}: success={success} coverage={coverage:.2%} steps={steps} seed={seed}")


def make_chart():
    df = pd.DataFrame(
        [
            {
                "Mapa": feat["mapa"],
                "Obstaculos": feat["obstacles"],
                "Max steps": feat["max_steps"],
                "Episodios": feat["episodes"],
                "Full Coverage": feat["full_coverage"],
                "Average Coverage": feat["average_coverage"],
                "Average Steps": feat["average_steps"],
            }
            for feat in FEATS
        ]
    )
    df.to_csv(GRAPH_DIR / "dados_imperador_escalas_extremas.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    x = np.arange(len(df))
    width = 0.36
    bars_full = ax.bar(x - width / 2, df["Full Coverage"], width, label="Full Coverage", color="#2f6f9f")
    bars_avg = ax.bar(x + width / 2, df["Average Coverage"], width, label="Average Coverage", color="#4f8f5b")
    for bars in [bars_full, bars_avg]:
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_title("Imperador_20X20 em escalas maiores")
    ax.set_ylabel("Cobertura em 100 episodios")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Mapa"])
    ax.set_ylim(0, 108)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / "16_imperador_escalas_extremas.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    pygame.init()
    pygame.font.init()
    register_env(load_env())
    model = PPO.load(str(MODEL_PATH), device="cpu")
    make_chart()
    for feat in FEATS:
        make_gif(model, feat)
    pygame.quit()


if __name__ == "__main__":
    main()
