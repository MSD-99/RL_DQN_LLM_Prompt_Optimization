import json
import random
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from env import BoolQEnv, load_data, load_llm, load_embedder
from env import KEEP, ADD_COT, NUM_ACTIONS
from dqn import DQNAgent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_method(env, action_fn, name, num_runs=1):
    all_sim = []
    all_tok = []
    all_rew = []

    for run in range(num_runs):
        desc = f"  {name}" + (f" run {run+1}" if num_runs > 1 else "")
        for idx in tqdm(range(len(env.samples)), desc=desc, leave=False):
            state = env.reset(idx=idx)
            ep_reward = 0.0

            for t in range(env.max_steps):
                action = action_fn(state, t)
                state, reward, done, info = env.step(action)
                ep_reward += reward

            all_sim.append(info['similarity'])
            all_tok.append(info['num_tokens'])
            all_rew.append(ep_reward)

    return {
        'name':           name,
        'avg_similarity': float(np.mean(all_sim)),
        'avg_tokens':     float(np.mean(all_tok)),
        'avg_reward':     float(np.mean(all_rew)),
        'std_reward':     float(np.std(all_rew)),
    }


def plot_training_curves(history, save_dir):
    """plot reward and loss curves from training history"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    ep_rewards = history['episode_rewards']
    avg_epoch = history['avg_rewards_per_epoch']
    losses = history.get('losses', [])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # (a) episode rewards + running avg
    ax = axes[0]
    ax.plot(ep_rewards, alpha=0.25, color='steelblue', linewidth=0.5)
    window = min(50, max(1, len(ep_rewards) // 10))
    if len(ep_rewards) >= window:
        running = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
        ax.plot(range(window-1, len(ep_rewards)), running,
                color='crimson', linewidth=2, label=f'Running avg (w={window})')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Reward')
    ax.set_title('Training Episode Rewards')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (b) avg per epoch
    ax = axes[1]
    ax.plot(range(1, len(avg_epoch)+1), avg_epoch, 'o-',
            color='seagreen', linewidth=2, markersize=4)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Average Reward')
    ax.set_title('Average Accumulated Reward per Epoch')
    ax.grid(True, alpha=0.3)

    # (c) loss
    ax = axes[2]
    if losses:
        ax.plot(losses, alpha=0.3, color='mediumpurple', linewidth=0.5)
        win = min(100, max(1, len(losses) // 20))
        if len(losses) >= win:
            loss_avg = np.convolve(losses, np.ones(win)/win, mode='valid')
            ax.plot(range(win-1, len(losses)), loss_avg,
                    color='darkred', linewidth=2, label=f'Running avg ({win})')
        ax.set_xlabel('Update Step')
        ax.set_ylabel('MSE Loss')
        ax.set_title('DQN Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No loss data', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)

    plt.tight_layout()
    path = Path(save_dir) / 'training_curves.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [plot] Saved: {path}")


def plot_test_comparison(results, save_dir):
    """bar chart of the 4 methods"""
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    names   = [r['name'] for r in results]
    rewards = [r['avg_reward'] for r in results]
    sims    = [r['avg_similarity'] for r in results]
    tokens  = [r['avg_tokens'] for r in results]
    colors  = ['#2196F3', '#FF9800', '#4CAF50', '#F44336']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, values, ylabel, title in [
        (axes[0], rewards, 'Average Reward',     'Test: Average Reward'),
        (axes[1], sims,    'Average Similarity',  'Test: Similarity (Accuracy)'),
        (axes[2], tokens,  'Average Tokens',      'Test: Average Tokens'),
    ]:
        bars = ax.bar(names, values, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        # put value on top of each bar
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + max(values)*0.02,
                    f'{v:.3f}', ha='center', va='bottom',
                    fontsize=10, fontweight='bold')

    plt.tight_layout()
    path = Path(save_dir) / 'test_comparison.png'
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [plot] Saved: {path}")


def print_table(results):
    header = f"{'Method':<10} {'Avg Similarity':>16} {'Avg Tokens':>12} {'Avg Reward':>13}"
    line = "=" * len(header)
    print(f"\n{line}\n{header}\n{line}")
    for r in results:
        print(f"{r['name']:<10} {r['avg_similarity']:>16.4f} "
              f"{r['avg_tokens']:>12.1f} {r['avg_reward']:>13.4f}")
    print(line)


def evaluate(args):
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[eval] Device: {device}")

    # Load data and models
    _, test_data = load_data(args.data_path)
    print(f"[eval] Test samples: {len(test_data)}")

    tokenizer, llm = load_llm(device)
    embedder = load_embedder()

    env = BoolQEnv(
        samples=test_data,
        tokenizer=tokenizer,
        model=llm,
        embedder=embedder,
        device=device,
        max_steps=2,
    )

    # Load trained DQN agent
    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        device=device,
    )
    ckpt_path = Path(args.model_dir) / 'dqn_agent.pth'
    agent.load(str(ckpt_path))
    agent.q_network.eval()
    print(f"[eval] Loaded DQN from {ckpt_path}")

    # --- run all 4 methods ---
    results = []

    # BASE - no modifiers
    print("\n>> BASE")
    results.append(evaluate_method(
        env, lambda s, t: KEEP, "BASE"))

    # COT - chain of thought at step 0
    print(">> COT")
    results.append(evaluate_method(
        env, lambda s, t: ADD_COT if t == 0 else KEEP, "COT"))

    # RANDOM - avg over 3 runs
    print(">> RANDOM")
    set_seed(args.seed)
    results.append(evaluate_method(
        env, lambda s, t: random.randint(0, NUM_ACTIONS - 1), "RANDOM", num_runs=3))

    # DQN - greedy
    print(">> DQN")
    results.append(evaluate_method(
        env, lambda s, t: agent.select_action(s, greedy=True), "DQN"))

    # --- results ---
    print_table(results)

    # Check: DQN should beat RANDOM
    dqn_reward = results[3]['avg_reward']
    random_reward = results[2]['avg_reward']
    if dqn_reward > random_reward:
        print(f"\n  DQN ({dqn_reward:.4f}) outperforms RANDOM ({random_reward:.4f})")
    else:
        print(f"\n  WARNING: DQN ({dqn_reward:.4f}) does NOT beat RANDOM ({random_reward:.4f})")
        print("  Try training for more epochs or tuning hyperparameters.")

    # --- plots ---

    # Training curves (from saved history)
    history_path = Path(args.model_dir) / 'training_history.json'
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plot_training_curves(history, args.results_dir)
    else:
        print("  [!!] training_history.json not found, skipping training plots")

    # test comparison chart
    plot_test_comparison(results, args.results_dir)

    # Save results as JSON
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    res_path = results_dir / 'test_results.json'
    res_path.write_text(json.dumps(results, indent=2))
    print(f"  [json] Saved: {res_path}")

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate DQN agent')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--data_path',   type=str, default='./data/boolq_200.json')
    parser.add_argument('--model_dir',   type=str, default='./saved_models')
    parser.add_argument('--results_dir', type=str, default='./results')
    args = parser.parse_args()

    evaluate(args)
