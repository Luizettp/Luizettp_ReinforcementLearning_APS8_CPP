from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


GRAPH_DIR = Path("Graficos")
GRAPH_DIR.mkdir(exist_ok=True)


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


COLORS = {
    "blue": "#2f6f9f",
    "green": "#4f8f5b",
    "orange": "#d88c2d",
    "red": "#b64b4b",
    "gray": "#6f6f6f",
}


def save_csv(df, name):
    df.to_csv(GRAPH_DIR / name, index=False)


def finish(fig, filename):
    fig.tight_layout()
    fig.savefig(GRAPH_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)


def label_bars(ax, bars, suffix="", fmt="{:.1f}"):
    for bar in bars:
        value = bar.get_height()
        if np.isnan(value):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{fmt.format(value)}{suffix}",
            ha="center",
            va="bottom",
            fontsize=9,
        )


entropy = pd.DataFrame(
    [
        {"ENTROPY_COEF": "0.05", "5x5": 93.0, "10x10 direto": 44.0},
        {"ENTROPY_COEF": "0.02", "5x5": 94.0, "10x10 direto": 50.0},
        {"ENTROPY_COEF": "0.007", "5x5": 95.0, "10x10 direto": 54.0},
        {"ENTROPY_COEF": "0.005", "5x5": 96.0, "10x10 direto": 41.0},
        {"ENTROPY_COEF": "0.00314", "5x5": 91.0, "10x10 direto": np.nan},
    ]
)
save_csv(entropy, "dados_01_entropia.csv")

fig, ax = plt.subplots(figsize=(9.5, 5.2))
x = np.arange(len(entropy))
width = 0.36
bars_5 = ax.bar(x - width / 2, entropy["5x5"], width, label="5x5", color=COLORS["blue"])
bars_10 = ax.bar(
    x + width / 2,
    entropy["10x10 direto"].fillna(0),
    width,
    label="10x10 direto",
    color=COLORS["orange"],
)
label_bars(ax, bars_5, "%")
label_bars(ax, bars_10, "%")
ax.set_title("Busca de entropia: exploração vs. fechamento da cobertura")
ax.set_ylabel("Full Coverage")
ax.set_xticks(x)
ax.set_xticklabels(entropy["ENTROPY_COEF"])
ax.set_xlabel("ENTROPY_COEF")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.25)
ax.legend()
finish(fig, "01_busca_entropia.png")


state_repr = pd.DataFrame(
    [
        {
            "Configuração": "Estado original",
            "Full Coverage 5x5": 95.0,
            "Full Coverage 10x10": 54.0,
            "Average Steps 10x10": 287.3,
        },
        {
            "Configuração": "Estado local enriquecido",
            "Full Coverage 5x5": 100.0,
            "Full Coverage 10x10": 68.0,
            "Average Steps 10x10": 243.2,
        },
    ]
)
save_csv(state_repr, "dados_02_estado_local.csv")

fig, ax = plt.subplots(figsize=(8.2, 5.0))
x = np.arange(len(state_repr))
bars_5 = ax.bar(
    x - width / 2,
    state_repr["Full Coverage 5x5"],
    width,
    label="5x5",
    color=COLORS["blue"],
)
bars_10 = ax.bar(
    x + width / 2,
    state_repr["Full Coverage 10x10"],
    width,
    label="10x10 direto",
    color=COLORS["green"],
)
label_bars(ax, bars_5, "%")
label_bars(ax, bars_10, "%")
ax.set_title("Representação local melhora aprendizado e transferência")
ax.set_ylabel("Full Coverage")
ax.set_xticks(x)
ax.set_xticklabels(state_repr["Configuração"])
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.25)
ax.legend()
finish(fig, "02_estado_local_enriquecido.png")


strategies = pd.DataFrame(
    [
        {
            "Estratégia": "Estado original\n5x5",
            "Full Coverage": 54.0,
            "Average Coverage": 97.70,
            "Average Steps": 287.3,
        },
        {
            "Estratégia": "Estado local\n5x5",
            "Full Coverage": 68.0,
            "Average Coverage": 99.24,
            "Average Steps": 243.2,
        },
        {
            "Estratégia": "Curriculum\n5x5 -> 8x8",
            "Full Coverage": 96.0,
            "Average Coverage": 99.96,
            "Average Steps": 161.8,
        },
        {
            "Estratégia": "Final 1000 ep.\n5x5 -> 8x8",
            "Full Coverage": 94.3,
            "Average Coverage": 99.86,
            "Average Steps": 182.8,
        },
        {
            "Estratégia": "Treino extra\n10x10 + 1M",
            "Full Coverage": 88.0,
            "Average Coverage": 99.49,
            "Average Steps": 208.0,
        },
    ]
)
save_csv(strategies, "dados_03_estrategias_10x10.csv")

fig, ax = plt.subplots(figsize=(9.5, 5.2))
bars = ax.bar(strategies["Estratégia"], strategies["Full Coverage"], color=COLORS["blue"])
label_bars(ax, bars, "%")
ax.set_title("Generalização no 10x10 por estratégia")
ax.set_ylabel("Full Coverage")
ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=0.25)
finish(fig, "03_full_coverage_10x10_por_estrategia.png")

fig, ax = plt.subplots(figsize=(9.5, 5.2))
bars = ax.bar(strategies["Estratégia"], strategies["Average Steps"], color=COLORS["orange"])
label_bars(ax, bars, fmt="{:.1f}")
ax.set_title("Eficiência no 10x10: menos passos para cobrir")
ax.set_ylabel("Average Steps")
ax.grid(axis="y", alpha=0.25)
finish(fig, "04_passos_medios_10x10_por_estrategia.png")


final_tests = pd.DataFrame(
    [
        {
            "Ambiente": "5x5",
            "Full Coverage": 98.90,
            "Average Coverage": 99.87,
            "Average Steps": 31.9,
            "Max Steps": 200,
        },
        {
            "Ambiente": "8x8",
            "Full Coverage": 95.90,
            "Average Coverage": 99.86,
            "Average Steps": 101.7,
            "Max Steps": 320,
        },
        {
            "Ambiente": "10x10",
            "Full Coverage": 94.30,
            "Average Coverage": 99.86,
            "Average Steps": 182.8,
            "Max Steps": 500,
        },
        {
            "Ambiente": "20x20",
            "Full Coverage": 36.70,
            "Average Coverage": 98.76,
            "Average Steps": 911.2,
            "Max Steps": 1000,
        },
    ]
)
save_csv(final_tests, "dados_04_testes_finais_1000.csv")

fig, ax = plt.subplots(figsize=(9.2, 5.2))
x = np.arange(len(final_tests))
bars_full = ax.bar(
    x - width / 2,
    final_tests["Full Coverage"],
    width,
    label="Full Coverage",
    color=COLORS["blue"],
)
bars_avg = ax.bar(
    x + width / 2,
    final_tests["Average Coverage"],
    width,
    label="Average Coverage",
    color=COLORS["green"],
)
label_bars(ax, bars_full, "%")
label_bars(ax, bars_avg, "%")
ax.set_title("Testes finais do PPO6_Possible_GOAT com 1000 episódios")
ax.set_ylabel("Cobertura")
ax.set_xticks(x)
ax.set_xticklabels(final_tests["Ambiente"])
ax.set_ylim(0, 110)
ax.grid(axis="y", alpha=0.25)
ax.legend()
finish(fig, "05_cobertura_final_1000_episodios.png")

fig, ax = plt.subplots(figsize=(9.2, 5.2))
x = np.arange(len(final_tests))
bars_avg = ax.bar(
    x - width / 2,
    final_tests["Average Steps"],
    width,
    label="Average Steps",
    color=COLORS["orange"],
)
bars_max = ax.bar(
    x + width / 2,
    final_tests["Max Steps"],
    width,
    label="Max Steps",
    color=COLORS["gray"],
)
label_bars(ax, bars_avg, fmt="{:.1f}")
label_bars(ax, bars_max, fmt="{:.0f}")
ax.set_title("Gargalo no 20x20: muitos episódios chegam perto do limite")
ax.set_ylabel("Passos")
ax.set_xticks(x)
ax.set_xticklabels(final_tests["Ambiente"])
ax.grid(axis="y", alpha=0.25)
ax.legend()
finish(fig, "06_passos_vs_limite_final.png")


def load_learning_curve(path, stage, offset=0):
    df = pd.read_csv(path)
    out = pd.DataFrame(
        {
            "timesteps": df["time/total_timesteps"].astype(float) + offset,
            "stage_timesteps": df["time/total_timesteps"].astype(float),
            "reward_mean": df["rollout/ep_rew_mean"].astype(float),
            "episode_len_mean": df["rollout/ep_len_mean"].astype(float),
            "stage": stage,
        }
    )
    out["reward_smooth"] = out["reward_mean"].rolling(9, min_periods=1).mean()
    out["episode_len_smooth"] = out["episode_len_mean"].rolling(9, min_periods=1).mean()
    return out


curve_5x5 = load_learning_curve(
    "log/ppo_cpp_5_3_200_0.007_20260506_094958/progress.csv",
    "Treino 5x5",
    offset=0,
)
curve_8x8 = load_learning_curve(
    "log/ppo_cpp_8_8_320_0.007_20260506_100925_curriculum/progress.csv",
    "Curriculum 8x8",
    offset=500_000,
)
learning = pd.concat([curve_5x5, curve_8x8], ignore_index=True)
save_csv(learning, "dados_05_curva_aprendizado_modelo_final.csv")

fig, ax = plt.subplots(figsize=(9.6, 5.2))
for stage, color in [("Treino 5x5", COLORS["blue"]), ("Curriculum 8x8", COLORS["green"])]:
    part = learning[learning["stage"] == stage]
    ax.plot(
        part["timesteps"],
        part["reward_smooth"],
        label=stage,
        color=color,
        linewidth=2.2,
    )
ax.axvline(500_000, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8)
ax.text(505_000, learning["reward_smooth"].min() + 5, "início do 8x8", fontsize=9)
ax.set_title("Curva de aprendizado do modelo final")
ax.set_xlabel("Timesteps acumulados")
ax.set_ylabel("rollout/ep_rew_mean suavizado")
ax.grid(axis="both", alpha=0.25)
ax.legend()
finish(fig, "07_curva_aprendizado_reward.png")

fig, ax = plt.subplots(figsize=(9.6, 5.2))
for stage, color in [("Treino 5x5", COLORS["blue"]), ("Curriculum 8x8", COLORS["green"])]:
    part = learning[learning["stage"] == stage]
    ax.plot(
        part["timesteps"],
        part["episode_len_smooth"],
        label=stage,
        color=color,
        linewidth=2.2,
    )
ax.axvline(500_000, color="#333333", linestyle="--", linewidth=1.2, alpha=0.8)
ax.text(505_000, learning["episode_len_smooth"].max() - 10, "início do 8x8", fontsize=9)
ax.set_title("Tamanho médio dos episódios durante o aprendizado")
ax.set_xlabel("Timesteps acumulados")
ax.set_ylabel("rollout/ep_len_mean suavizado")
ax.grid(axis="both", alpha=0.25)
ax.legend()
finish(fig, "08_curva_aprendizado_tamanho_episodio.png")

print("Gráficos finais gerados em Graficos/.")
