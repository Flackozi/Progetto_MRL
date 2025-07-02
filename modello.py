import lunarLanderClasses as LCC
import gymnasium as gym
import numpy as np

def main():
    env = gym.make('LunarLander-v3')
    agent = LCC.LunarLanderClass(
        numEpisodes=0,  # non serve ri-addestrare
        Alpha=0,        # parametri non rilevanti per test
        Epsilon=0.0,
        Lambda=0,
        Gamma=0,
        k = 0
    )
    agent.initStage(stage=1)
    agent.load_policy("policy_sarsa.npy")
    print("✅ Policy caricata e pronta per il test!")

    avg_reward = agent.test_policy(env, render=True, num_episodes=5)
    print(f"🎯 Reward medio su 5 ep = {avg_reward:.2f}")
    env.close()

if __name__ == "__main__":
    main()