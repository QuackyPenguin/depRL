import gym
import sconegym
import deprl


# create the sconegym env
env = gym.make("sconerun_BWR_twoRew_org_h0918addAbd-v1")  #"sconerunInputVel_torso_h0918addAbd_gaps-v0") #sconeruntest_h0918addAbd-v0") 

policy = deprl.load("/media/calc_2/scone_results/live/denis/curr_BWAR_5tasks_oneRew_org_LastTask04_envTrans06/241018.232030.H0918v2j_abbAdd/", env, 1.5e7)

for episode in range(10):
    # # store the results of every 10th episode
    # # storing results is slow, and should only be done sparsely
    # # stored results can be analyzed in SCONE Studio
    if episode % 1 == 0:
        env.store_next_episode()

    episode_steps = 0
    total_reward = 0
    # print("RESET")
    state = env.reset()

    while True:
        # samples random action
        action = policy(state) #, deterministic=True)

        # applies action and advances environment by one step
        next_state, reward, done, info = env.step(action)

        episode_steps += 1
        total_reward += reward

        # to render results, open a .sto file in SCONE Studio
        # env.render()
        state = next_state

        # check if done
        if done or (episode_steps >= 1400): #1000
            #print(total_reward)
            #print(env.rand_target_vel)
            print(
                f"Episode {episode} finished; steps={episode_steps}; reward={total_reward:0.3f}"
            )
            env.write_now()
            episode += 1
            break

env.close()
