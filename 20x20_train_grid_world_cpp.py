#
# python 20x20_train_grid_world_cpp.py <train|test|run|curriculum> dim obstacles max_steps total_timesteps
#

# Uso:
#
# Treinar:
#   python 20x20_train_grid_world_cpp.py train 20 48 1500 50000
#
# Testar modelo treinado:
#   python 20x20_train_grid_world_cpp.py test 20 48 1500 1000 data/20x20_experiments/MERLIN_2020.zip
#
# Visualizar modelo treinado:
#   python 20x20_train_grid_world_cpp.py run 20 48 1500 data/20x20_experiments/MERLIN_2020.zip
#
# Continuar treinamento a partir de um modelo pré-treinado:
#   python 20x20_train_grid_world_cpp.py curriculum 20 48 1500 50000 data/20x20_experiments/MERLIN_2020.zip
#
# Argumentos:
#   mode             = train, test, run ou curriculum
#   dim              = tamanho do grid. Ex: 5, 10, 20
#   obstacles        = quantidade de obstáculos. Ex: 3, 12, 48
#   max_steps        = máximo de passos por episódio. Ex: 200, 500, 1000
#   total_timesteps  = quantidade de passos de treino. Usado só em train/curriculum, 500_000
#   CPP_PROFILE=gladiador  -> local_history_v2, reward original
#   CPP_PROFILE=merlin     -> local_memory_decay_v2, rastro direcional curto
#   CPP_PROFILE=imperador  -> local_memory_v1, memória de células observadas
#   Se CPP_PROFILE não for definido, este script experimental usa merlin.

import gymnasium as gym
import importlib.util
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from datetime import datetime
import sys  
import os
from pathlib import Path

BEST_BASELINE_MODEL = "ppo20_local_history_v2_original_policy128_8_8_320_0.003_20260507_030154_curriculum"
BEST_BASELINE_TRAINING = "5x5 500000 timesteps -> 8x8 500000 timesteps"
BEST_BASELINE_RESULT_20X20 = "51.00% full coverage over 200 episodes at 1000 max_steps; 63.00% at 1600 max_steps"

PROFILE = os.environ.get("CPP_PROFILE", "merlin").lower()
PROFILE_DEFAULTS = {
    "gladiador": {
        "state": "local_history_v2",
        "reward": "original",
        "n_steps": "4096",
        "gamma": "0.995",
        "learning_rate": "0.0003",
        "batch_size": "128",
    },
    "merlin": {
        "state": "local_memory_decay_v2",
        "reward": "late_finish_v1",
        "n_steps": "4096",
        "gamma": "0.995",
        "learning_rate": "0.00001",
        "batch_size": "128",
        "memory_decay": "0.92",
        "memory_trace_gain": "0.40",
    },
    "imperador": {
        "state": "local_memory_v1",
        "reward": "original",
        "n_steps": "4096",
        "gamma": "0.995",
        "learning_rate": "0.00003",
        "batch_size": "128",
    },
}
PROFILE_CONFIG = PROFILE_DEFAULTS.get(PROFILE, PROFILE_DEFAULTS["merlin"])
if "memory_decay" in PROFILE_CONFIG:
    os.environ.setdefault("CPP_MEMORY_DECAY", PROFILE_CONFIG["memory_decay"])
if "memory_trace_gain" in PROFILE_CONFIG:
    os.environ.setdefault("CPP_MEMORY_TRACE_GAIN", PROFILE_CONFIG["memory_trace_gain"])

DEFAULT_ALGO = "ppo"
DEFAULT_POLICY_ARCH = "128,128"
DEFAULT_STATE_VARIANT = PROFILE_CONFIG["state"]
DEFAULT_REWARD_VARIANT = PROFILE_CONFIG["reward"]
DEFAULT_ENTROPY_COEF = "0.003"
DEFAULT_LOCAL_VIEW_SIZE = "3"
DEFAULT_N_STEPS = PROFILE_CONFIG["n_steps"]
DEFAULT_GAMMA = PROFILE_CONFIG["gamma"]
DEFAULT_GAE_LAMBDA = "0.95"
DEFAULT_LEARNING_RATE = PROFILE_CONFIG["learning_rate"]
DEFAULT_BATCH_SIZE = PROFILE_CONFIG["batch_size"]
DEFAULT_N_EPOCHS = "10"
DEFAULT_CLIP_RANGE = "0.2"

ALGO = os.environ.get("CPP_ALGO", DEFAULT_ALGO).lower()
IS_RECURRENT = ALGO in {"recurrent_ppo", "recurrentppo", "rppo"}
POLICY_ARCH = [int(layer_size) for layer_size in os.environ.get("CPP_POLICY_ARCH", DEFAULT_POLICY_ARCH).split(",")]
POLICY_ARCH_LABEL = str(POLICY_ARCH[0]) if len(POLICY_ARCH) == 2 and POLICY_ARCH[0] == POLICY_ARCH[1] else "x".join(str(layer_size) for layer_size in POLICY_ARCH)
STATE_VARIANT = os.environ.get("CPP_STATE_VARIANT", DEFAULT_STATE_VARIANT)
REWARD_VARIANT = os.environ.get("CPP_REWARD_VARIANT", DEFAULT_REWARD_VARIANT)
LOCAL_VIEW_SIZE = int(os.environ.get("CPP_LOCAL_VIEW_SIZE", DEFAULT_LOCAL_VIEW_SIZE))
VERBOSE = int(os.environ.get("CPP_VERBOSE", "1"))
N_STEPS = int(os.environ.get("CPP_N_STEPS", DEFAULT_N_STEPS))
CHECKPOINT_FREQ = int(os.environ.get("CPP_CHECKPOINT_FREQ", "100000"))
DETERMINISTIC_TEST = os.environ.get("CPP_DETERMINISTIC_TEST", "0") == "1"
PRINT_TEST_EPISODES = os.environ.get("CPP_PRINT_TEST_EPISODES", "1") == "1"
ENTROPY_COEF = float(os.environ.get("CPP_ENTROPY_COEF", DEFAULT_ENTROPY_COEF))
GAMMA = float(os.environ.get("CPP_GAMMA", DEFAULT_GAMMA))
GAE_LAMBDA = float(os.environ.get("CPP_GAE_LAMBDA", DEFAULT_GAE_LAMBDA))
LEARNING_RATE = float(os.environ.get("CPP_LEARNING_RATE", DEFAULT_LEARNING_RATE))
BATCH_SIZE = int(os.environ.get("CPP_BATCH_SIZE", DEFAULT_BATCH_SIZE))
N_EPOCHS = int(os.environ.get("CPP_N_EPOCHS", DEFAULT_N_EPOCHS))
CLIP_RANGE = float(os.environ.get("CPP_CLIP_RANGE", DEFAULT_CLIP_RANGE))
LOG_FORMATS = [item.strip() for item in os.environ.get("CPP_LOG_FORMATS", "stdout,csv,tensorboard").split(",") if item.strip()]


def label_float(value: float) -> str:
    return f"{value:g}".replace(".", "p")


HP_LABEL = f"n{N_STEPS}_g{label_float(GAMMA)}_lr{label_float(LEARNING_RATE)}"

if IS_RECURRENT:
    from sb3_contrib import RecurrentPPO

    MODEL_CLASS = RecurrentPPO
    POLICY_NAME = "MultiInputLstmPolicy"
    EXPERIMENT_NAME = f"ppo20_{STATE_VARIANT}_{REWARD_VARIANT}_view{LOCAL_VIEW_SIZE}_{HP_LABEL}_recurrent"
    EXPERIMENT_DESCRIPTION = f"RecurrentPPO MultiInputLstmPolicy, ent_coef={ENTROPY_COEF}, gamma={GAMMA}, gae_lambda={GAE_LAMBDA}, learning_rate={LEARNING_RATE}, n_steps={N_STEPS}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}, clip_range={CLIP_RANGE}, state={STATE_VARIANT}, reward={REWARD_VARIANT}, local_view={LOCAL_VIEW_SIZE}x{LOCAL_VIEW_SIZE}"
elif ALGO in {"maskable_ppo", "maskableppo", "mppo"}:
    from sb3_contrib import MaskablePPO
    from sb3_contrib.common.maskable.utils import get_action_masks

    MODEL_CLASS = MaskablePPO
    POLICY_NAME = "MultiInputPolicy"
    EXPERIMENT_NAME = f"ppo20_{STATE_VARIANT}_{REWARD_VARIANT}_view{LOCAL_VIEW_SIZE}_{HP_LABEL}_maskable_policy{POLICY_ARCH_LABEL}"
    EXPERIMENT_DESCRIPTION = f"MaskablePPO MultiInputPolicy, ent_coef={ENTROPY_COEF}, gamma={GAMMA}, gae_lambda={GAE_LAMBDA}, learning_rate={LEARNING_RATE}, n_steps={N_STEPS}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}, clip_range={CLIP_RANGE}, net_arch pi/vf={POLICY_ARCH}, state={STATE_VARIANT}, reward={REWARD_VARIANT}, local_view={LOCAL_VIEW_SIZE}x{LOCAL_VIEW_SIZE}"
else:
    MODEL_CLASS = PPO
    POLICY_NAME = "MultiInputPolicy"
    EXPERIMENT_NAME = f"ppo20_{STATE_VARIANT}_{REWARD_VARIANT}_view{LOCAL_VIEW_SIZE}_{HP_LABEL}_policy{POLICY_ARCH_LABEL}"
    EXPERIMENT_DESCRIPTION = f"PPO MultiInputPolicy, ent_coef={ENTROPY_COEF}, gamma={GAMMA}, gae_lambda={GAE_LAMBDA}, learning_rate={LEARNING_RATE}, n_steps={N_STEPS}, batch_size={BATCH_SIZE}, n_epochs={N_EPOCHS}, clip_range={CLIP_RANGE}, net_arch pi/vf={POLICY_ARCH}, state={STATE_VARIANT}, reward={REWARD_VARIANT}, local_view={LOCAL_VIEW_SIZE}x{LOCAL_VIEW_SIZE}"
EXPERIMENT_ENV_ID = "gymnasium_env/GridWorldCPP20x20-v0"
EXPERIMENT_DATA_DIR = Path("data") / "20x20_experiments"
EXPERIMENT_LOG_DIR = Path("log") / "20x20_experiments"
RESULTS_PATH = Path("results_20x20_experiments.md")

POLICY_KWARGS = dict(
    net_arch=dict(
        pi=POLICY_ARCH,
        vf=POLICY_ARCH,
    )
)


def _load_experimental_env():
    """Load 20x20_grid_world_cpp.py despite the filename starting with a digit."""
    env_path = Path(__file__).with_name("20x20_grid_world_cpp.py")
    spec = importlib.util.spec_from_file_location("grid_world_cpp_20x20_experiment", env_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load experimental environment from {env_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.GridWorldCPPEnv


GridWorldCPPEnv = _load_experimental_env()

def print_action(action: int) -> str:
    return {
        0: "right",
        1: "up",
        2: "left",
        3: "down",
    }.get(action, "unknown")


def predict_action(model, obs, deterministic=False, lstm_states=None, episode_starts=None):
    if IS_RECURRENT:
        action, lstm_states = model.predict(
            obs,
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=deterministic,
        )
        return action, lstm_states

    if MODEL_CLASS.__name__ == "MaskablePPO":
        action_masks = get_action_masks(model.get_env())
        action, _ = model.predict(
            obs,
            deterministic=deterministic,
            action_masks=action_masks,
        )
        return action, lstm_states

    action, _ = model.predict(obs, deterministic=deterministic)
    return action, lstm_states

# --- Argument validation ---
if len(sys.argv) < 2 or sys.argv[1] not in ['train', 'test', 'run', 'curriculum']:
    print("Usage: python 20x20_train_grid_world_cpp.py <train|test|run|curriculum> dim obstacles max_steps [total_timesteps|num_episodes] [model_name]")
    sys.exit(1)

mode = sys.argv[1]

if mode in ['train', 'curriculum']:
    valid_lengths = [6] if mode == 'train' else [6, 7]
    if len(sys.argv) not in valid_lengths:
        print("Usage for training: python 20x20_train_grid_world_cpp.py train|curriculum dim obstacles max_steps total_timesteps [model_name]")
        sys.exit(1)

elif mode == 'test':
    if len(sys.argv) not in [5, 6, 7]:
        print("Usage for testing: python 20x20_train_grid_world_cpp.py test dim obstacles max_steps [num_episodes] [model_name]")
        sys.exit(1)

elif mode == 'run':
    if len(sys.argv) not in [5, 6]:
        print("Usage for running: python 20x20_train_grid_world_cpp.py run dim obstacles max_steps [model_name]")
        sys.exit(1)

# --- Hyperparameters ---
DIM = int(sys.argv[2])            # 5, 10, 20
OBSTACLES = int(sys.argv[3])      # 3, 12, 48
MAX_STEPS = int(sys.argv[4])      # 200, 400, 800

if mode in ['train', 'curriculum']:
    TOTAL_TIMESTEPS = int(sys.argv[5])  # 500_000, 1_000_000
else:
    TOTAL_TIMESTEPS = None              # Not used in test/run

# -----------------------

EXPERIMENT_DATA_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENT_LOG_DIR.mkdir(parents=True, exist_ok=True)


def resolve_model_path(model_name: str) -> Path:
    model_name = model_name if model_name.endswith(".zip") else f"{model_name}.zip"
    raw_path = Path(model_name)

    if raw_path.is_absolute() or raw_path.parent != Path("."):
        return raw_path

    experimental_path = EXPERIMENT_DATA_DIR / raw_path.name
    legacy_path = Path("data") / raw_path.name

    if experimental_path.exists():
        return experimental_path
    if legacy_path.exists():
        return legacy_path

    return experimental_path


def get_model_name(arg_index: int) -> str:
    if len(sys.argv) > arg_index:
        return sys.argv[arg_index]

    env_model_name = os.environ.get("CPP_MODEL_NAME")
    if env_model_name:
        return env_model_name

    return input("Enter model filename (without .zip, or a path): ")


def append_test_result(
    model_name,
    num_episodes,
    full_coverage_count,
    full_coverage_rate,
    avg_coverage,
    std_coverage,
    min_coverage,
    max_coverage,
    avg_steps,
    std_steps,
    min_steps,
    max_steps_observed,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not RESULTS_PATH.exists():
        RESULTS_PATH.write_text(
            "# 20x20 Experiments\n\n"
            "Arquivo separado para a linha experimental 20x20. Os scripts principais nao foram alterados.\n\n",
            encoding="utf-8",
        )

    with RESULTS_PATH.open("a", encoding="utf-8") as file:
        file.write(f"## {timestamp} - {EXPERIMENT_NAME}\n\n")
        file.write(f"- Model: `{model_name}`\n")
        file.write(f"- Strategy: {EXPERIMENT_DESCRIPTION}\n")
        file.write("- State: partial observation with normalized position, coverage_ratio, neighbors 3x3, local ratios, and local/history features\n")
        file.write(f"- Test grid: `{DIM}x{DIM}`, obstacles `{OBSTACLES}`, max_steps `{MAX_STEPS}`\n")
        file.write(f"- Episodes: `{num_episodes}`\n")
        file.write(f"- Deterministic policy: `{DETERMINISTIC_TEST}`\n")
        file.write(f"- Full Coverage Rate: `{full_coverage_rate:.2f}% ({full_coverage_count}/{num_episodes})`\n")
        file.write(f"- Average Coverage: `{avg_coverage:.2f}%`\n")
        file.write(f"- Std Coverage: `{std_coverage:.2f}%`\n")
        file.write(f"- Min Coverage: `{min_coverage:.2f}%`\n")
        file.write(f"- Max Coverage: `{max_coverage:.2f}%`\n")
        file.write(f"- Average Steps: `{avg_steps:.1f}`\n")
        file.write(f"- Std Steps: `{std_steps:.1f}`\n")
        file.write(f"- Min Steps: `{min_steps}`\n")
        file.write(f"- Max Steps: `{max_steps_observed}`\n\n")


def make_checkpoint_callback(log_dir):
    if CHECKPOINT_FREQ <= 0:
        return None

    checkpoint_dir = Path(log_dir) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    return CheckpointCallback(
        save_freq=CHECKPOINT_FREQ,
        save_path=str(checkpoint_dir),
        name_prefix="ckpt",
    )

# -----------------------

try:
    gym.register(
        id=EXPERIMENT_ENV_ID,
        entry_point=GridWorldCPPEnv,
    )
except Exception:
    pass



if mode == 'train':
    print("--- Starting CPP Training ---")
    env = gym.make(
        EXPERIMENT_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )
    check_env(env)

    model_kwargs = {}
    if not IS_RECURRENT:
        model_kwargs["policy_kwargs"] = POLICY_KWARGS

    model = MODEL_CLASS(
        POLICY_NAME,
        env,
        verbose=VERBOSE,
        ent_coef=ENTROPY_COEF,
        device="cpu",
        n_steps=N_STEPS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        clip_range=CLIP_RANGE,
        **model_kwargs,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = EXPERIMENT_LOG_DIR / f'{EXPERIMENT_NAME}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}'
    model_path = EXPERIMENT_DATA_DIR / f'{EXPERIMENT_NAME}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}.zip'
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    new_logger = configure(str(log_dir), LOG_FORMATS)
    model.set_logger(new_logger)

    print(f"Starting learning with {TOTAL_TIMESTEPS} timesteps...")
    checkpoint_callback = make_checkpoint_callback(log_dir)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == 'curriculum':

    print("--- Starting CPP Curriculum Learning Training ---")
    
    model_name = get_model_name(6)
    model_path = resolve_model_path(model_name)

    env = gym.make(        
        EXPERIMENT_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )

    # Carrega os pesos do modelo 5x5 e associa ao novo ambiente
    model = MODEL_CLASS.load(
        str(model_path),
        env=env,
        device="cpu",
        n_steps=N_STEPS,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        learning_rate=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        n_epochs=N_EPOCHS,
        clip_range=CLIP_RANGE,
    )

    # Continua o treinamento com os pesos já inicializados

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = EXPERIMENT_LOG_DIR / f'{EXPERIMENT_NAME}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}_curriculum'
    model_path = EXPERIMENT_DATA_DIR / f'{EXPERIMENT_NAME}_{DIM}_{OBSTACLES}_{MAX_STEPS}_{ENTROPY_COEF}_{timestamp}_curriculum.zip'
    log_dir.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    new_logger = configure(str(log_dir), LOG_FORMATS)
    model.set_logger(new_logger)

    print(f"Starting learning with {TOTAL_TIMESTEPS} timesteps...")
    checkpoint_callback = make_checkpoint_callback(log_dir)
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
    model.save(model_path)
    print(f"Model trained and saved to {model_path}")
    print(f"Logs saved to {log_dir}")

elif mode == 'run':
    model_name = get_model_name(5)
    model_path = resolve_model_path(model_name)
    print(f'--- Loading model from {model_path} for a run ---')

    env = gym.make(
        EXPERIMENT_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="human"
    )
    model = MODEL_CLASS.load(str(model_path), env=env, device="cpu")

    (obs, info) = env.reset()
    done = False
    truncated = False
    steps = 0
    total_reward = 0
    lstm_states = None
    episode_starts = np.ones((1,), dtype=bool)
    while not done and not truncated:
        action, lstm_states = predict_action(
            model,
            obs,
            deterministic=True,
            lstm_states=lstm_states,
            episode_starts=episode_starts,
        )
        obs, reward, done, truncated, info = env.step(action.item())
        episode_starts = np.array([done or truncated], dtype=bool)
        total_reward += reward
        steps += 1
        print(f"Step: {steps}, Action: {print_action(action.item())}, "
              f"Reward: {reward:.2f}, Coverage: {info['coverage']:.1%}, "
              f"Done: {done}, Truncated: {truncated}")
    print(f"--- Run Finished --- Total reward: {total_reward:.2f}, Coverage: {info['coverage']:.1%}")

elif mode == 'test':
    if len(sys.argv) == 6 and not sys.argv[5].isdigit():
        num_episodes = int(os.environ.get("CPP_TEST_EPISODES", "1000"))
        model_name = sys.argv[5]
    else:
        num_episodes = int(sys.argv[5]) if len(sys.argv) > 5 else int(os.environ.get("CPP_TEST_EPISODES", "1000"))
        model_name = get_model_name(6)

    model_path = resolve_model_path(model_name)
    print(f'--- Loading model from {model_path} for testing ---')

    env = gym.make(
        EXPERIMENT_ENV_ID,
        size=DIM,
        obs_quantity=OBSTACLES,
        max_steps=MAX_STEPS,
        render_mode="rgb_array"
    )
    model = MODEL_CLASS.load(str(model_path), env=env, device="cpu")

    full_coverage_count = 0
    total_coverages = []
    total_steps_list = []

    for i in range(num_episodes):
        (obs, info) = env.reset()
        done = False
        truncated = False
        steps = 0
        lstm_states = None
        episode_starts = np.ones((1,), dtype=bool)
        while not done and not truncated:
            action, lstm_states = predict_action(
                model,
                obs,
                deterministic=DETERMINISTIC_TEST,
                lstm_states=lstm_states,
                episode_starts=episode_starts,
            )
            obs, reward, done, truncated, info = env.step(action.item())
            episode_starts = np.array([done or truncated], dtype=bool)
            steps += 1

        total_coverages.append(info['coverage'])
        total_steps_list.append(steps)

        if done and not truncated:
            full_coverage_count += 1
            if PRINT_TEST_EPISODES:
                print(f"Episode {i+1}: Full coverage in {steps} steps.")
        else:
            if PRINT_TEST_EPISODES:
                print(f"Episode {i+1}: Coverage {info['coverage']:.1%} in {steps} steps.")

    full_coverage_rate = (full_coverage_count / num_episodes) * 100
    avg_coverage = np.mean(total_coverages) * 100
    standard_deviation = np.std(total_coverages) * 100
    avg_steps = np.mean(total_steps_list)
    standard_deviation_steps = np.std(total_steps_list)
    print(f"\n--- Test Finished ---")
    print(f"Full Coverage Rate: {full_coverage_rate:.2f}% ({full_coverage_count}/{num_episodes})")
    print(f"Average Coverage: {avg_coverage:.2f}% Standard Deviation: {standard_deviation:.2f}% Min Coverage: {np.min(total_coverages)*100:.2f}% Max Coverage: {np.max(total_coverages)*100:.2f}%")
    print(f"Average Steps: {avg_steps:.1f} Standard Deviation: {standard_deviation_steps:.1f} Min Steps: {np.min(total_steps_list)} Max Steps: {np.max(total_steps_list)}")
    append_test_result(
        model_name=model_path.name,
        num_episodes=num_episodes,
        full_coverage_count=full_coverage_count,
        full_coverage_rate=full_coverage_rate,
        avg_coverage=avg_coverage,
        std_coverage=standard_deviation,
        min_coverage=np.min(total_coverages) * 100,
        max_coverage=np.max(total_coverages) * 100,
        avg_steps=avg_steps,
        std_steps=standard_deviation_steps,
        min_steps=np.min(total_steps_list),
        max_steps_observed=np.max(total_steps_list),
    )
