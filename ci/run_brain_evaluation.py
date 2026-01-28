from __future__ import annotations
import os
import sys
import json
from typing import Tuple, Dict, Any, List
import numpy as np

LIB_DIR = os.path.join(os.path.dirname(__file__), "..", "contextual_bandit_brain")
REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")
os.makedirs(ARTIFACTS_DIR, exist_ok=True)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from contextual_bandit_brain.core.brain import LinUCBBrain
from contextual_bandit_brain.simulator.environment import Environment
from contextual_bandit_brain.bis.metrics import (
    compute_reward_score,
    compute_regret_efficiency,
    compute_stability,
    compute_adaptability,
    compute_fairness,
)
from contextual_bandit_brain.bis.scoring import compute_bis
from contextual_bandit_brain.reporting.report_generator import write_json_report, generate_plots, write_text_summary


def expected_reward(theta: np.ndarray, x: np.ndarray) -> float:
    raw = float(x.dot(theta))
    return 1.0 / (1.0 + np.exp(-raw))


def sampler(rng: np.random.Generator, d: int) -> np.ndarray:
    return rng.normal(0.0, 1.0, size=d).astype(float)


def run_episode(env: Environment, brain: LinUCBBrain, steps: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rewards = []
    regrets = []
    chosen = []
    exploration_flags = []
    contexts = []
    for _ in range(steps):
        x = sampler(env._rng, env.d)
        a = brain.select_action(x)
        info = brain.explain_last()
        exploration_flags.append(info["mode"] == "exploration")
        r = env.reward(a, x)
        brain.update(a, r, x)
        rewards.append(r)
        exp_rewards = [expected_reward(env.theta()[i], x) for i in range(env.num_actions)]
        regrets.append(max(exp_rewards) - exp_rewards[a])
        chosen.append(a)
        contexts.append(x)
    return (
        np.asarray(rewards, dtype=float),
        np.asarray(regrets, dtype=float),
        np.asarray(chosen, dtype=int),
        np.asarray(exploration_flags, dtype=bool),
        np.asarray(contexts, dtype=float),
    )


def main() -> int:
    d = int(os.environ.get("LINUCB_D", "8"))
    num_actions = int(os.environ.get("LINUCB_ACTIONS", "5"))
    alpha = float(os.environ.get("LINUCB_ALPHA", "0.5"))
    steps1 = int(os.environ.get("LINUCB_STEPS1", "2000"))
    steps2 = int(os.environ.get("LINUCB_STEPS2", "2000"))
    seeds = [0, 1]

    metrics_runs = []
    for seed in seeds:
        env = Environment(d=d, num_actions=num_actions, noise_std=0.1, seed=seed)
        brain = LinUCBBrain(num_actions=num_actions, alpha=alpha, d=d)
        rewards1, regrets1, chosen1, exploration1, contexts1 = run_episode(env, brain, steps1)
        env.drift(scale=0.6)
        rewards2, regrets2, chosen2, exploration2, contexts2 = run_episode(env, brain, steps2)
        best_exp_sum = float(np.sum([max([expected_reward(env.theta()[i], x) for i in range(env.num_actions)]) for x in contexts1]))
        reward_score = compute_reward_score(rewards1, best_exp_sum)
        regret_eff = compute_regret_efficiency(regrets1, best_exp_sum)
        stability = compute_stability(chosen1, env.num_actions, window=max(100, steps1 // 10))
        conv_win = max(50, steps2 // 20)
        rolling_regret = np.convolve(regrets2, np.ones(conv_win) / conv_win, mode="valid")
        conv_step = steps2
        for idx, val in enumerate(rolling_regret):
            if val <= 0.05:
                conv_step = idx + conv_win - 1
                break
        adaptability = compute_adaptability(rewards2, np.asarray([max([expected_reward(env.theta()[i], x) for i in range(env.num_actions)]) for x in contexts2]), conv_step, steps2)
        fairness = compute_fairness(np.concatenate([chosen1, chosen2]), env.num_actions)
        exploration_ratio = float(np.mean(np.concatenate([exploration1, exploration2]).astype(float)))
        metrics_runs.append({
            "reward": reward_score,
            "regret": regret_eff,
            "stability": stability,
            "adaptability": adaptability,
            "fairness": fairness,
            "exploration_ratio": exploration_ratio,
        })
        if seed == seeds[0]:
            generate_plots(
                out_dir=ARTIFACTS_DIR,
                rewards=np.concatenate([rewards1, rewards2]),
                regrets=np.concatenate([regrets1, regrets2]),
                exploration_flags=np.concatenate([exploration1, exploration2]),
                chosen_actions=np.concatenate([chosen1, chosen2]),
                bis_score=0.0,
            )
    agg = {key: float(np.mean([m[key] for m in metrics_runs])) for key in metrics_runs[0].keys()}
    bis = compute_bis({"reward": agg["reward"], "regret": agg["regret"], "stability": agg["stability"], "adaptability": agg["adaptability"], "fairness": agg["fairness"]})
    write_json_report(ARTIFACTS_DIR, bis_score=bis, metrics=agg)
    write_text_summary(ARTIFACTS_DIR, bis_score=bis, metrics=agg)
    threshold = float(os.environ.get("BIS_THRESHOLD", "0.75"))
    return 0 if bis >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
