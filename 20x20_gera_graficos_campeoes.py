from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRAPH_DIR = Path("Graficos")
GRAPH_DIR.mkdir(exist_ok=True)

COLORS = {
    "gladiador": "#9b5a2e",
    "merlin": "#5f5b9f",
    "imperador": "#2f6f9f",
    "gray": "#6f6f6f",
    "green": "#4f8f5b",
}

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#222222",
        "axes.titleweight": "bold",
        "axes.titlesize": 13,
        "axes.labelsize": 10,
        "xtick.color": "#222222",
        "ytick.color": "#222222",
        "font.size": 10,
        "legend.frameon": False,
    }
)


def label_bars(ax, bars, suffix="%", fmt="{:.1f}"):
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{fmt.format(value)}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def finish(fig, filename):
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


champions = pd.DataFrame(
    [
        {
            "Modelo": "Gladiador_2020",
            "Ambiente": "5x5",
            "Max steps": 200,
            "Full Coverage": 99.90,
            "Average Coverage": 100.00,
            "Average Steps": 28.4,
        },
        {
            "Modelo": "Gladiador_2020",
            "Ambiente": "10x10",
            "Max steps": 500,
            "Full Coverage": 98.90,
            "Average Coverage": 99.98,
            "Average Steps": 151.7,
        },
        {
            "Modelo": "Gladiador_2020",
            "Ambiente": "20x20",
            "Max steps": 2000,
            "Full Coverage": 87.70,
            "Average Coverage": 99.92,
            "Average Steps": 1067.8,
        },
        {
            "Modelo": "Gladiador_2020",
            "Ambiente": "20x20",
            "Max steps": 2400,
            "Full Coverage": 88.40,
            "Average Coverage": 99.93,
            "Average Steps": 1143.2,
        },
        {
            "Modelo": "Imperador_20X20",
            "Ambiente": "5x5",
            "Max steps": 200,
            "Full Coverage": 100.00,
            "Average Coverage": 100.00,
            "Average Steps": 25.3,
        },
        {
            "Modelo": "MERLIN_2020",
            "Ambiente": "5x5",
            "Max steps": 200,
            "Full Coverage": 99.90,
            "Average Coverage": 100.00,
            "Average Steps": 29.3,
        },
        {
            "Modelo": "MERLIN_2020",
            "Ambiente": "10x10",
            "Max steps": 500,
            "Full Coverage": 99.10,
            "Average Coverage": 99.94,
            "Average Steps": 141.4,
        },
        {
            "Modelo": "MERLIN_2020",
            "Ambiente": "20x20",
            "Max steps": 1000,
            "Full Coverage": 75.20,
            "Average Coverage": 99.78,
            "Average Steps": 747.2,
        },
        {
            "Modelo": "MERLIN_2020",
            "Ambiente": "20x20",
            "Max steps": 1500,
            "Full Coverage": 90.30,
            "Average Coverage": 99.95,
            "Average Steps": 820.4,
        },
        {
            "Modelo": "Imperador_20X20",
            "Ambiente": "10x10",
            "Max steps": 500,
            "Full Coverage": 99.60,
            "Average Coverage": 99.99,
            "Average Steps": 110.7,
        },
        {
            "Modelo": "Imperador_20X20",
            "Ambiente": "20x20",
            "Max steps": 1000,
            "Full Coverage": 98.00,
            "Average Coverage": 99.98,
            "Average Steps": 510.2,
        },
        {
            "Modelo": "Imperador_20X20",
            "Ambiente": "20x20",
            "Max steps": 1500,
            "Full Coverage": 99.80,
            "Average Coverage": 100.00,
            "Average Steps": 507.4,
        },
    ]
)
champions.to_csv(GRAPH_DIR / "dados_20x20_campeoes.csv", index=False)

compare_20 = champions[champions["Ambiente"] == "20x20"].copy()
compare_20["Label"] = compare_20["Modelo"].str.replace("_", "\n") + "\n" + compare_20["Max steps"].astype(str) + " steps"

fig, ax = plt.subplots(figsize=(9.0, 5.0))
colors = [
    COLORS["gladiador"] if model == "Gladiador_2020"
    else COLORS["merlin"] if model == "MERLIN_2020"
    else COLORS["imperador"]
    for model in compare_20["Modelo"]
]
bars = ax.bar(compare_20["Label"], compare_20["Full Coverage"], color=colors)
label_bars(ax, bars)
ax.axhline(90, color="#333333", linestyle="--", linewidth=1.1, alpha=0.75)
ax.text(-0.45, 91.5, "meta 90%", fontsize=9)
ax.set_title("20x20: tres formas de atacar a cauda final")
ax.set_ylabel("Full Coverage em 1000 episodios")
ax.set_ylim(0, 108)
ax.grid(axis="y", alpha=0.25)
finish(fig, "12_campeoes_20x20_resultados.png")

scale = champions[
    ((champions["Modelo"] == "Gladiador_2020") & (champions["Max steps"].isin([200, 500, 2400])))
    | ((champions["Modelo"] == "MERLIN_2020") & (champions["Max steps"].isin([200, 500, 1500])))
    | ((champions["Modelo"] == "Imperador_20X20") & (champions["Max steps"].isin([200, 500, 1000])))
].copy()
order = ["5x5", "10x10", "20x20"]
fig, ax = plt.subplots(figsize=(9.2, 5.0))
x = np.arange(len(order))
width = 0.25
glad = [scale[(scale["Modelo"] == "Gladiador_2020") & (scale["Ambiente"] == env)]["Full Coverage"].iloc[0] for env in order]
mer = [scale[(scale["Modelo"] == "MERLIN_2020") & (scale["Ambiente"] == env)]["Full Coverage"].iloc[0] for env in order]
imp = [scale[(scale["Modelo"] == "Imperador_20X20") & (scale["Ambiente"] == env)]["Full Coverage"].iloc[0] for env in order]
bars_g = ax.bar(x - width, glad, width, label="Gladiador_2020", color=COLORS["gladiador"])
bars_m = ax.bar(x, mer, width, label="MERLIN_2020", color=COLORS["merlin"])
bars_i = ax.bar(x + width, imp, width, label="Imperador_20X20", color=COLORS["imperador"])
label_bars(ax, bars_g)
label_bars(ax, bars_m)
label_bars(ax, bars_i)
ax.set_title("Generalizacao dos caminhos 20x20 em 1000 episodios")
ax.set_ylabel("Full Coverage")
ax.set_xticks(x)
ax.set_xticklabels(order)
ax.set_ylim(0, 108)
ax.grid(axis="y", alpha=0.25)
ax.legend()
finish(fig, "13_campeoes_20x20_escalas.png")


def load_curve(path, label, offset):
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "timesteps": df["time/total_timesteps"].astype(float) + offset,
            "episode_len_mean": df["rollout/ep_len_mean"].astype(float),
            "reward_mean": df["rollout/ep_rew_mean"].astype(float),
            "stage": label,
        }
    )
    out["episode_len_smooth"] = out["episode_len_mean"].rolling(5, min_periods=1).mean()
    out["reward_smooth"] = out["reward_mean"].rolling(5, min_periods=1).mean()
    return out


curves = pd.concat(
    [
        load_curve(
            "log/20x20_experiments/ppo20_local_history_v2_original_view3_n4096_g0p995_lr0p0003_policy128_15_27_750_0.003_20260507_144522_curriculum/progress.csv",
            "Gladiador: adaptacao 15x15",
            0,
        ),
        load_curve(
            "log/20x20_experiments/ppo20_local_memory_v1_original_view3_n4096_g0p995_lr3e-05_policy128_20_48_1000_0.003_20260507_222956_curriculum/progress.csv",
            "Imperador: fine-tune 20x20",
            0,
        ),
        load_curve(
            "log/20x20_experiments/ppo20_local_memory_decay_v2_late_finish_v1_view3_n4096_g0p995_lr1e-05_policy128_20_48_1500_0.003_20260508_185040_curriculum/progress.csv",
            "MERLIN: fine-tune 20x20",
            0,
        ),
    ],
    ignore_index=True,
)
curves.to_csv(GRAPH_DIR / "dados_20x20_curvas_campeoes.csv", index=False)

fig, ax = plt.subplots(figsize=(9.2, 5.0))
for stage, color in [
    ("Gladiador: adaptacao 15x15", COLORS["gladiador"]),
    ("MERLIN: fine-tune 20x20", COLORS["merlin"]),
    ("Imperador: fine-tune 20x20", COLORS["imperador"]),
]:
    part = curves[curves["stage"] == stage]
    ax.plot(part["timesteps"], part["episode_len_smooth"], label=stage, color=color, linewidth=2.2)
ax.set_title("Curva de aprendizado dos ajustes finais")
ax.set_xlabel("Timesteps do estagio")
ax.set_ylabel("rollout/ep_len_mean suavizado")
ax.grid(axis="both", alpha=0.25)
ax.legend()
finish(fig, "14_campeoes_20x20_curva_aprendizado.png")

print("Graficos dos campeoes 20x20 gerados.")
