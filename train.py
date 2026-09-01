import json
import random
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from tqdm import tqdm

from env import BoolQEnv, load_data, load_llm, load_embedder
from dqn import DQNAgent


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args):
    set_seed(args.seed)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    print(f"[train] Device: {device}")
    print(f"[train] Seed:   {args.seed}")

    # Load data and models
    train_data, _ = load_data(args.data_path)
    print(f"[train] Training samples: {len(train_data)}")

    tokenizer, llm = load_llm(device)
    embedder = load_embedder()

    # Create environment
    env = BoolQEnv(
        samples=train_data,
        tokenizer=tokenizer,
        model=llm,
        embedder=embedder,
        device=device,
        max_steps=2,
    )

    agent = DQNAgent(
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=1.0,
        epsilon_end=0.05,
        epsilon_decay=0.9,
        buffer_capacity=10000,
        batch_size=args.batch_size,
        target_update_freq=args.target_update,
        device=device,
    )

    # Create save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # training
    all_episode_rewards = []
    avg_rewards_per_epoch = []

    print(f"\n{'=' * 60}")
    print(f"  Training: {args.epochs} epochs x {len(train_data)} episodes (T=2)")
    print(f"{'=' * 60}\n")

    for epoch in range(1, args.epochs + 1):
        epoch_rewards = []
        indices = list(range(len(train_data)))
        random.shuffle(indices)

        pbar = tqdm(indices, desc=f"Epoch {epoch:02d}/{args.epochs}", leave=False)
        for idx in pbar:
            state = env.reset(idx=idx)
            episode_reward = 0.0

            for t in range(env.max_steps):
                action = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                agent.store_transition(state, action, reward, next_state, done)
                agent.update()
                state = next_state
                episode_reward += reward

            all_episode_rewards.append(episode_reward)
            epoch_rewards.append(episode_reward)
            pbar.set_postfix(r=f"{episode_reward:+.3f}", eps=f"{agent.epsilon:.3f}")

        agent.decay_epsilon()

        # log
        avg_r = float(np.mean(epoch_rewards))
        avg_rewards_per_epoch.append(avg_r)
        last_loss = agent.losses[-1] if agent.losses else float('nan')
        print(f"  Epoch {epoch:02d}/{args.epochs}  |  "
              f"Avg reward: {avg_r:+.4f}  |  "
              f"Epsilon: {agent.epsilon:.4f}  |  "
              f"Loss: {last_loss:.6f}  |  "
              f"Buffer: {len(agent.replay_buffer)}")

    # save model
    model_path = save_dir / 'dqn_agent.pth'
    agent.save(str(model_path))

    # Save training history
    history = {
        'episode_rewards':       all_episode_rewards,
        'avg_rewards_per_epoch': avg_rewards_per_epoch,
        'losses':                agent.losses,
        'seed':                  args.seed,
        'epochs':                args.epochs,
        'lr':                    args.lr,
        'gamma':                 args.gamma,
        'batch_size':            args.batch_size,
        'target_update_freq':    args.target_update,
        'timestamp':             datetime.now().isoformat(),
    }
    history_path = save_dir / 'training_history.json'
    history_path.write_text(json.dumps(history, indent=2))

    print(f"\n  Model saved   -> {model_path}")
    print(f"  History saved -> {history_path}")
    print(f"  Done.\n")
    return agent, history


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train DQN agent')
    parser.add_argument('--epochs',        type=int,   default=40)
    parser.add_argument('--lr',            type=float, default=1e-3)
    parser.add_argument('--gamma',         type=float, default=0.99)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--target_update', type=int,   default=10)
    parser.add_argument('--seed',          type=int,   default=42)
    parser.add_argument('--data_path',     type=str,   default='./data/boolq_200.json')
    parser.add_argument('--save_dir',      type=str,   default='./saved_models')
    args = parser.parse_args()

    train(args)
