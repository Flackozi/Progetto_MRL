import lunarLanderClasses as LCC
import gymnasium as gym
import numpy as np

def main():
    env = gym.make('LunarLander-v3')
    agent = LCC.LunarLanderClass(
        numEpisodes=0,  # non serve ri-addestrare
        Alpha=0,        # parametri non rilevanti per test
        Epsilon=0.0,    # IMPORTANTE: epsilon = 0 per policy greedy
        Lambda=0,
        Gamma=0,
        k=0
    )
    agent.initStage(stage=1)
    
    # Prova prima il nuovo formato
    try:
        agent.load_policy("policy_sarsa_complete.pkl")
    except:
        # Se non esiste, usa il vecchio formato
        print("Provo a caricare il formato vecchio...")
        agent.load_policy("policy_sarsa.npy")
    
    print("✅ Policy caricata e pronta per il test!")

    # Test più esteso per avere statistiche migliori
    avg_reward = agent.test_policy(env, render=True, num_episodes=10)
    print(f"🎯 Reward medio su 10 episodi = {avg_reward:.2f}")
    
    # Test senza rendering per statistiche più veloci
    print("\nTest senza rendering (20 episodi)...")
    avg_reward_fast = agent.test_policy(env, render=False, num_episodes=20)
    print(f"🎯 Reward medio su 20 episodi = {avg_reward_fast:.2f}")
    
    env.close()

if __name__ == "__main__":
    main()