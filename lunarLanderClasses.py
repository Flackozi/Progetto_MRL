import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym
from tiles3 import IHT, tiles
from tqdm import trange
import pickle


class TileCoder:
    def __init__(self, iht_size=4096, num_tilings=8, tiles_per_dim=8, state_bounds=None):
        self.iht = IHT(iht_size)
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim

        # Rimuovi leg1 e leg2 -> solo primi 6 elementi
        self.state_bounds = state_bounds or [
            (-1.5, 1.5),     # x position
            (0, 1.5),        # y position  
            (-3, 3),         # x velocity
            (-3, 3),         # y velocity
            (-np.pi, np.pi), # angle
            (-8, 8)          # angular velocity
        ]

        self.num_dims = len(self.state_bounds)

    def get_tiles(self, state):
        # Considera solo le prime 6 dimensioni (esclude leg1, leg2)
        scaled_state = []
        for i in range(self.num_dims):
            min_val, max_val = self.state_bounds[i]
            ratio = (state[i] - min_val) / (max_val - min_val)
            scaled_val = ratio * self.tiles_per_dim
            scaled_state.append(scaled_val)
        return tiles(self.iht, self.num_tilings, scaled_state)

class LunarLanderClass:
    def __init__(self, numEpisodes, Alpha, Epsilon, Lambda, Gamma, k):
        """
        Inizializza la classe per l'apprendimento SARSA(λ) su Lunar Lander con Tile Coding
        """
        self.numEpisodes = numEpisodes
        self.alpha = Alpha
        self.epsilon = Epsilon
        self.lambda_param = Lambda
        self.gamma = Gamma
        self.k = k
        
        # Spazio delle azioni per Lunar Lander (4 azioni discrete)
        self.num_actions = 4
        
        # Inizializza il tile coder con parametri ottimizzati
        self.tile_coder = TileCoder(
            iht_size=65536,
            num_tilings=8,
            tiles_per_dim=8,
            state_bounds=[
                (-1.5, 1.5),     # x
                (0, 1.5),        # y
                (-3, 3),         # vx
                (-3, 3),         # vy
                (-np.pi, np.pi), # angle
                (-8, 8)          # angular vel
            ]
        )
        
        # Pesi e eligibility traces per ogni feature
        self.weights = defaultdict(float)
        self.eligibility_traces = defaultdict(float)
        
        # Statistiche per il monitoraggio
        self.episode_rewards = []
        self.episode_lengths = []
        
        # Parametri inizializzati
        self.initialized = False
        
    def initStage(self, stage):
        """Inizializza lo stage di apprendimento"""
        self.stage = stage
        self.initialized = True
        print(f"Inizializzato stage {stage} con Tile Coding")
        
    def get_active_tiles(self, state):
        """Ottiene i tile attivi per lo stato continuo"""
        return self.tile_coder.get_tiles(state)
    
    def get_state_action_features(self, state, action):
        """Crea le feature per la coppia stato-azione usando tile coding"""
        active_tiles = self.get_active_tiles(state[:6])  # Solo le 6 dimensioni continue
        leg1, leg2 = int(state[6]), int(state[7])        # Binari
        features = []
        for tile in active_tiles:
            features.append((tile, action, leg1, leg2))
        return features

    def epsilon_greedy(self, state, episode):
        """Implementa la strategia epsilon-greedy per la selezione delle azioni"""
        # Durante il test, usa sempre policy greedy
        if self.epsilon == 0.0 or np.random.random() >= self.epsilon:
            # Azione greedy (migliore Q-value)
            q_values = []
            for action in range(self.num_actions):
                q_value = self.get_q_value(state, action)
                q_values.append(q_value)
            return np.argmax(q_values)
        else:
            # Azione casuale
            return np.random.randint(0, self.num_actions)
    
    def get_q_value(self, state, action):
        """Ottiene il Q-value per una coppia stato-azione usando tile coding"""
        features = self.get_state_action_features(state, action)
        q_value = 0.0
        for feature in features:
            q_value += self.weights[feature]
        return q_value
    
    def update_eligibility_traces(self, state, action):
        """Aggiorna le eligibility traces per tile coding"""
        # Applica il decadimento a tutte le traces esistenti
        for feature in list(self.eligibility_traces.keys()):
            self.eligibility_traces[feature] *= self.gamma * self.lambda_param
        
        # Imposta le traces per le feature attive correnti
        features = self.get_state_action_features(state, action)
        for feature in features:
            old_value = self.eligibility_traces.get(feature, 0.0)
            self.eligibility_traces[feature] = (1 - self.alpha) * old_value + 1
    
    def SARSALambda(self, env, render_episode_interval=None):
        """Implementa l'algoritmo SARSA(λ) per Lunar Lander con Tile Coding"""
        if not self.initialized:
            raise Exception("Devi chiamare initStage() prima di eseguire SARSA(λ)")
        
        print("Iniziando l'addestramento SARSA(λ) con Tile Coding...")
        
        for episode in trange(self.numEpisodes, desc="Addestramento SARSA(λ)", ncols=100):
            # Reset dell'ambiente
            state, _ = env.reset()
            
            # Selezione azione iniziale
            action = self.epsilon_greedy(state, episode)
            
            episode_reward = 0
            episode_length = 0
            
            # Reset eligibility traces per ogni episodio
            self.eligibility_traces.clear()
            
            while True:
                # Esegui l'azione
                next_state, reward, terminated, truncated, _ = env.step(action)
                
                episode_reward += reward
                episode_length += 1
                
                # Calcola Q(s,a)
                q_current = self.get_q_value(state, action)
                
                if terminated or truncated:
                    # Stato terminale: Q(s',a') = 0
                    td_error = reward - q_current
                    
                    # Aggiorna eligibility traces
                    self.update_eligibility_traces(state, action)
                    
                    # Aggiorna tutti i pesi usando le eligibility traces
                    for feature, trace in self.eligibility_traces.items():
                        self.weights[feature] += self.alpha * td_error * trace
                    
                    break
                else:
                    # Seleziona prossima azione
                    next_action = self.epsilon_greedy(next_state, episode)
                    
                    # Calcola Q(s',a')
                    q_next = self.get_q_value(next_state, next_action)
                    
                    # Calcola TD error
                    td_error = reward + self.gamma * q_next - q_current
                    
                    # Aggiorna eligibility traces
                    self.update_eligibility_traces(state, action)
                    
                    # Aggiorna tutti i pesi usando le eligibility traces
                    for feature, trace in self.eligibility_traces.items():
                        self.weights[feature] += self.alpha * td_error * trace
                    
                    # Passa al prossimo stato-azione
                    state = next_state
                    action = next_action
            
            # Salva statistiche
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            
            # Stampa progresso
            if episode % 10000 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episodio {episode}, Reward medio (ultimi 100): {avg_reward:.2f}")
            
            # Test della policy
            if render_episode_interval and episode % render_episode_interval == 0 and episode > 0:
                print(f"\n--- Test Policy all'episodio {episode} ---")
                avg_test_reward = self.test_policy(env, render=True, num_episodes=1)
                print(f"Reward medio nel test: {avg_test_reward:.2f}")
                print("--- Fine Test ---\n")
        
        print("Addestramento completato!")
        self.save_policy("policy_sarsa_complete.pkl")
        self.plot_training_stats()
    
    def test_policy(self, env, render=False, num_episodes=1, max_steps=500):
        """Testa la policy appresa"""
        total_rewards = []
        
        # Crea un ambiente separato per il rendering se necessario
        if render:
            test_env = gym.make('LunarLander-v3', render_mode='human')
        else:
            test_env = env
        
        for episode in range(num_episodes):
            if render:
                state, _ = test_env.reset()
            else:
                state, _ = env.reset()
            
            episode_reward = 0
            steps = 0  
            
            while steps < max_steps: 
                # Usa policy greedy (epsilon = 0)
                q_values = []
                for act in range(self.num_actions):
                    q_values.append(self.get_q_value(state, act))
                action = np.argmax(q_values)
                
                if render:
                    next_state, reward, terminated, truncated, _ = test_env.step(action)
                else:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                
                episode_reward += reward
                steps += 1

                if terminated or truncated:
                    break
                
                state = next_state
            
            total_rewards.append(episode_reward)
            if render:
                print(f"Episodio test {episode + 1}: Reward = {episode_reward:.2f}, Passi = {steps}")

        if render:
            test_env.close()
        
        avg_reward = np.mean(total_rewards)
        if not render:
            print(f"Test completato - Reward medio: {avg_reward:.2f}")
        return avg_reward
    
    def plot_training_stats(self):
        """Plotta le statistiche di addestramento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot rewards
        ax1.plot(self.episode_rewards)
        ax1.set_title('Reward per Episodio (Tile Coding)')
        ax1.set_xlabel('Episodio')
        ax1.set_ylabel('Reward')
        
        # Plot running average
        window_size = min(100, len(self.episode_rewards) // 10)
        if window_size > 1:
            running_avg = np.convolve(self.episode_rewards, 
                                    np.ones(window_size)/window_size, mode='valid')
            ax1.plot(range(window_size-1, len(self.episode_rewards)), 
                    running_avg, 'r-', label=f'Media mobile ({window_size})')
            ax1.legend()
        
        # Plot episode lengths
        ax2.plot(self.episode_lengths)
        ax2.set_title('Lunghezza Episodi')
        ax2.set_xlabel('Episodio')
        ax2.set_ylabel('Passi')
        
        plt.tight_layout()
        plt.show()
    
    def save_policy(self, filename):
        """Salva la policy completa incluso il tile coder"""
        policy_data = {
            'weights': dict(self.weights),
            'tile_coder_state': {
                'iht_size': self.tile_coder.iht.size,
                'num_tilings': self.tile_coder.num_tilings,
                'tiles_per_dim': self.tile_coder.tiles_per_dim,
                'state_bounds': self.tile_coder.state_bounds,
                'iht_dictionary': self.tile_coder.iht.dictionary,
                'iht_count': self.tile_coder.iht.count
            },
            'training_params': {
                'alpha': self.alpha,
                'epsilon': self.epsilon,
                'lambda_param': self.lambda_param,
                'gamma': self.gamma,
                'k': self.k
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(policy_data, f)
        print(f"Policy completa salvata in {filename}")
    
    def load_policy(self, filename):
        """Carica una policy completa"""
        try:
            with open(filename, 'rb') as f:
                policy_data = pickle.load(f)
            
            # Ripristina i weights
            self.weights = defaultdict(float, policy_data['weights'])
            
            # Ripristina il tile coder
            tile_state = policy_data['tile_coder_state']
            self.tile_coder = TileCoder(
                iht_size=tile_state['iht_size'],
                num_tilings=tile_state['num_tilings'],
                tiles_per_dim=tile_state['tiles_per_dim'],
                state_bounds=tile_state['state_bounds']
            )
            
            # Ripristina lo stato interno dell'IHT
            self.tile_coder.iht.dictionary = tile_state['iht_dictionary']
            self.tile_coder.iht.count = tile_state['iht_count']
            
            # Ripristina i parametri di training (opzionale, per riferimento)
            params = policy_data['training_params']
            print(f"Policy caricata da {filename}")
            print(f"Parametri originali: α={params['alpha']}, ε={params['epsilon']}, λ={params['lambda_param']}")
            
        except FileNotFoundError:
            print(f"File {filename} non trovato. Provo a caricare il formato vecchio...")
            self.load_policy_old_format(filename.replace('.pkl', '.npy'))
        except Exception as e:
            print(f"Errore nel caricamento: {e}")
            print("Provo a caricare il formato vecchio...")
            self.load_policy_old_format(filename.replace('.pkl', '.npy'))
    
    def load_policy_old_format(self, filename):
        """Carica una policy nel formato vecchio (solo per compatibilità)"""
        try:
            loaded_weights = np.load(filename, allow_pickle=True).item()
            self.weights = defaultdict(float, loaded_weights)
            print(f"Policy caricata da {filename} (formato vecchio)")
            print("ATTENZIONE: Il tile coder potrebbe non essere identico a quello usato durante l'addestramento!")
        except Exception as e:
            print(f"Errore nel caricamento del formato vecchio: {e}")