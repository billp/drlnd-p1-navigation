"""Command-line training entry point for the Navigation (Banana) DQN project.

Equivalent to the training loop in Navigation.ipynb, for users who prefer to run
from the terminal instead of a notebook.

Usage:
    python train.py --env_path Banana.app
    python train.py --env_path /data/Banana_Linux_NoVis/Banana.x86_64 --n_episodes 2000
"""
import argparse
from collections import deque

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from unityagents import UnityEnvironment
from dqn_agent import Agent


def dqn(agent, env, brain_name, n_episodes=2000, max_t=1000,
        eps_start=1.0, eps_end=0.01, eps_decay=0.995, solve_score=13.0):
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        state = env_info.vector_observations[0]
        score = 0
        for _ in range(max_t):
            action = int(agent.act(state, eps))
            env_info = env.step(action)[brain_name]
            next_state = env_info.vector_observations[0]
            reward = env_info.rewards[0]
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print("\rEpisode {}\tAverage Score: {:.2f}".format(i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= solve_score:
            print("\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}".format(
                i_episode - 100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), "checkpoint.pth")
            break
    return scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_path", required=True, help="Path to the Unity Banana executable")
    parser.add_argument("--n_episodes", type=int, default=2000)
    args = parser.parse_args()

    env = UnityEnvironment(file_name=args.env_path)
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    env_info = env.reset(train_mode=True)[brain_name]
    state_size = len(env_info.vector_observations[0])
    action_size = brain.vector_action_space_size
    print("state_size={}  action_size={}".format(state_size, action_size))

    agent = Agent(state_size=state_size, action_size=action_size, seed=0)
    scores = dqn(agent, env, brain_name, n_episodes=args.n_episodes)
    env.close()

    plt.figure()
    plt.plot(np.arange(len(scores)), scores)
    plt.axhline(y=13.0, color="r", linestyle="--", label="solved (+13)")
    plt.ylabel("Score")
    plt.xlabel("Episode #")
    plt.legend()
    plt.savefig("assets/scores.png", dpi=120, bbox_inches="tight")
    print("Saved score plot to assets/scores.png")


if __name__ == "__main__":
    main()
