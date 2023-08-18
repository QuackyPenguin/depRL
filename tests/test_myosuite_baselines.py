import gym
import myosuite  # noqa
import numpy as np
import torch

import deprl

torch.set_default_device("cpu")

SEED = 1


def helper_env_loop(env):
    policy = deprl.load_baseline(env)
    policy.noisy = False
    returns = []
    qpos = []
    for ep in range(1):
        ret = 0
        env.seed(SEED)
        obs = env.reset()
        for i in range(200):
            action = policy(obs)
            obs, reward, done, _ = env.step(action)
            # env.sim.renderer.render_to_window()
            ret += reward
            qpos.append(env.sim.data.qpos[1])
            if done:
                break
        returns.append(ret)
    env.close()
    return returns, qpos


def test_myolegwalk():
    name = "myoLegWalk-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    torch.manual_seed(SEED)
    returns, qpos = helper_env_loop(env)
    assert np.round(np.mean(qpos), 2) == -1.47
    # assert np.floor(returns[0]) == 3511


def test_chasetag():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="init")
    env.seed(SEED)
    returns, qpos = helper_env_loop(env)
    print(np.mean(qpos))
    # assert np.mean(qpos) < -1.5


def test_relocate():
    name = "myoChallengeRelocateP1-v0"
    env = gym.make(name)
    env.seed(SEED)
    torch.manual_seed(SEED)
    returns, _ = helper_env_loop(env)
    # assert np.abs(np.floor(returns[0])) == 7538


def test_rng():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    policies = []
    for i in range(3):
        env.reset()
        policies.append(env.opponent.opponent_policy)
    # for x, y in zip(policies, ['repeller', 'random', 'random']):
    #     assert(x == y)


def test_rng_noise():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="random")
    env.seed(SEED)
    for i in range(3):
        env.reset()
        for i in range(5):
            env.opponent.noise_process.sample()
    # assert not (np.mean(noise) + 1.3004040323) > 1e-6


def test_chasetag_obs_rng():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="init")
    env.seed(SEED)
    env.reset()
    for i in range(20):
        old_obs, *_ = env.step(env.np_random.normal(0, 1, size=(80,)))

    diff = 0
    for i in range(100):
        env.seed(SEED)
        obs = env.reset()
        for i in range(20):
            obs, *_ = env.step(env.np_random.normal(0, 1, size=(80,)))
        diff += np.abs(old_obs - obs)
    print(f"{diff=}")


def test_chasetag_actionrng():
    name = "myoChallengeChaseTagP1-v0"
    env = gym.make(name, reset_type="init")
    env.seed(SEED)
    env.reset()
    for i in range(20):
        old_obs, *_ = env.step(env.np_random.normal(0, 1, size=(80,)))

    policy = deprl.load_baseline(env)
    init_action = policy(old_obs)
    diff = 0
    for i in range(100):
        action = policy(old_obs)
        diff += np.abs(action - init_action)
    print(diff)


if __name__ == "__main__":
    # test_chasetag()
    # test_chasetag_actionrng()
    # test_chasetag_obs_rng()
    # test_relocate()
    # test_rng_noise()
    test_myolegwalk()
    # test_relocate()