import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

class TileCoder:
    """
    Implementa il tile coding per la rappresentazione degli stati continui
    """
    def __init__(self, num_tilings=8, tiles_per_dim=8, state_bounds=None):
        """
        Args:
            num_tilings: Numero di tiling sovrapposti
            tiles_per_dim: Numero di tile per dimensione
            state_bounds: Limiti delle dimensioni dello stato [(min, max), ...]
        """
        self.num_tilings = num_tilings
        self.tiles_per_dim = tiles_per_dim
        self.state_bounds = state_bounds or [
            (-1.5, 1.5),    # x position
            (0, 1.5),       # y position  
            (-2, 2),        # x velocity
            (-2, 2),        # y velocity
            (-np.pi, np.pi), # angle
            (-5, 5),        # angular velocity
            (0, 1),         # leg1 contact
            (0, 1)          # leg2 contact
        ]
    
        
        self.num_dims = len(self.state_bounds)
        
        # Calcola gli offset per ogni tiling
        self.offsets = []
        for i in range(self.num_tilings):
            offset = []
            for dim in range(self.num_dims):
                range_size = self.state_bounds[dim][1] - self.state_bounds[dim][0]
                tile_width = range_size / self.tiles_per_dim
                # Offset uniforme per ogni tiling
                offset_val = (i / self.num_tilings) * tile_width
                offset.append(offset_val)
            self.offsets.append(offset)
    
    def get_tiles(self, state):
        """
        Restituisce gli indici dei tile attivi per lo stato dato
        
        Args:
            state: Stato continuo (array numpy)
            
        Returns:
            list: Lista degli indici dei tile attivi
        """
        active_tiles = []
        
        for tiling in range(self.num_tilings):
            tile_coords = []
            
            for dim in range(self.num_dims):
                # Applica i bounds e l'offset per questo tiling
                val = state[dim]
                min_val, max_val = self.state_bounds[dim]
                
                # Clamp del valore nei bounds
                val = max(min_val, min(max_val, val))
                
                # Calcola la coordinata del tile con offset
                range_size = max_val - min_val
                tile_width = range_size / self.tiles_per_dim
                
                # Applica l'offset specifico per questo tiling
                shifted_val = val - min_val + self.offsets[tiling][dim]
                tile_coord = int(shifted_val / tile_width) # --> per ottenere la coordinata del tile in questa dimensione
                
                # Assicura che sia nei limiti
                tile_coord = max(0, min(self.tiles_per_dim - 1, tile_coord))
                tile_coords.append(tile_coord)
            
            # Calcola l'indice univoco per questo tile (tipo sub2ind)
            tile_index = 0
            multiplier = 1
            for coord in reversed(tile_coords):
                tile_index += coord * multiplier
                multiplier *= self.tiles_per_dim
            
            # Aggiunge l'offset del tiling per renderlo unico
            tile_index += tiling * (self.tiles_per_dim ** self.num_dims)
            active_tiles.append(tile_index)
        
        return active_tiles

class LunarLanderClass:
    def __init__(self, numEpisodes, Alpha, initialEpsilon, Lambda, Gamma, k, epsUpdate):
        """
        Inizializza la classe per l'apprendimento SARSA(λ) su Lunar Lander con Tile Coding
        
        Args:
            numEpisodes: Numero di episodi di training
            Alpha: Learning rate
            initialEpsilon: Valore iniziale di epsilon per epsilon-greedy
            Lambda: Parametro lambda per eligibility traces
            Gamma: Fattore di sconto
            k: Parametro per il decadimento di epsilon
        """
        self.numEpisodes = numEpisodes
        self.alpha = Alpha
        self.epsilon = initialEpsilon
        self.lambda_param = Lambda
        self.gamma = Gamma
        self.k = k
        self.epsUpdate = epsUpdate
        
        # Spazio delle azioni per Lunar Lander (4 azioni discrete)
        self.num_actions = 4
        
        # Inizializza il tile coder con parametri ottimizzati
        self.tile_coder = TileCoder(
            num_tilings=8,
            tiles_per_dim=10,  # Aumentato per maggiore risoluzione
            state_bounds=[
                (-1.5, 1.5),     # x position
                (0, 1.5),        # y position  
                (-3, 3),         # x velocity - range ampliato
                (-3, 3),         # y velocity - range ampliato
                (-np.pi, np.pi), # angle
                (-8, 8),         # angular velocity - range ampliato
                (0, 1),          # leg1 contact
                (0, 1)           # leg2 contact
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
        """
        Ottiene i tile attivi per lo stato continuo
        
        Args:
            state: Stato continuo dall'ambiente (8-dimensional array)
            
        Returns:
            list: Lista degli indici dei tile attivi
        """
        return self.tile_coder.get_tiles(state)
    
    def get_state_action_features(self, state, action):
        """
        Crea le feature per la coppia stato-azione usando tile coding
        
        Args:
            state: Stato continuo
            action: Azione
            
        Returns:
            list: Lista delle feature attive per questa coppia stato-azione
        """
        active_tiles = self.get_active_tiles(state)
        # Combina i tile con l'azione per creare feature uniche
        features = []
        for tile in active_tiles:
            # Crea una feature unica per ogni combinazione tile-azione
            feature = (tile, action)
            features.append(feature)
        return features
    
    def epsilon_greedy(self, state, episode):
        """
        Implementa la strategia epsilon-greedy per la selezione delle azioni
        
        Args:
            state: Stato continuo
            episode: Numero dell'episodio corrente
            
        Returns:
            int: Azione selezionata
        """
        # Calcola epsilon con decadimento per migliorare l'apprendimento
        self.epsilon = self.epsilon - self.epsUpdate
        self.epsilon = max(0.05, self.epsilon)  # Minimo epsilon per mantenere esplorazione
        
        if np.random.random() < self.epsilon:
            # Azione casuale
            return np.random.randint(0, self.num_actions)
        else:
            # Azione greedy (migliore Q-value)
            q_values = []
            for action in range(self.num_actions):
                q_value = self.get_q_value(state, action)
                q_values.append(q_value)
            
            # Restituisce l'azione con il Q-value più alto
            return np.argmax(q_values)
    
    def get_q_value(self, state, action):
        """
        Ottiene il Q-value per una coppia stato-azione usando tile coding
        
        Args:
            state: Stato continuo
            action: Azione
            
        Returns:
            float: Q-value come somma dei pesi delle feature attive
        """
        features = self.get_state_action_features(state, action)
        q_value = 0.0
        for feature in features:
            q_value += self.weights[feature]
        return q_value
    
    def update_eligibility_traces(self, state, action):
        """
        Aggiorna le eligibility traces per tile coding
        
        Args:
            state: Stato continuo
            action: Azione
        """
        # Applica il decadimento a tutte le traces esistenti
        for feature in list(self.eligibility_traces.keys()):
            self.eligibility_traces[feature] *= self.gamma * self.lambda_param
            if abs(self.eligibility_traces[feature]) < 1e-8:
                del self.eligibility_traces[feature]
        
        # Imposta le traces per le feature attive correnti
        features = self.get_state_action_features(state, action)
        for feature in features:
            # Per tile coding, ogni feature attiva ottiene trace = 1
            self.eligibility_traces[feature] = 1.0
    
    def SARSALambda(self, env, render_episode_interval=None):
        """
        Implementa l'algoritmo SARSA(λ) per Lunar Lander con Tile Coding
        
        Args:
            env: Ambiente Gymnasium
            render_episode_interval: Intervallo per il rendering (None = mai)
        """
        if not self.initialized:
            raise Exception("Devi chiamare initStage() prima di eseguire SARSA(λ)")
        
        print("Iniziando l'addestramento SARSA(λ) con Tile Coding...")
        
        for episode in range(self.numEpisodes):
            # Reset dell'ambiente
            state, _ = env.reset()
            # Nota: ora usiamo direttamente lo stato continuo, senza discretizzazione
            
            # Selezione azione iniziale
            action = self.epsilon_greedy(state, episode)
            
            episode_reward = 0
            episode_length = 0
            
            # Reset eligibility traces per ogni episodio
            self.eligibility_traces.clear()
            
            while True:
                # Esegui l'azione
                next_state, reward, terminated, truncated, _ = env.step(action)
                # Nota: anche qui usiamo direttamente lo stato continuo
                
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
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episodio {episode}, Reward medio (ultimi 100): {avg_reward:.2f}")
            
            # Test della policy
            if render_episode_interval and episode % render_episode_interval == 0 and episode > 0:
                print(f"\n--- Test Policy all'episodio {episode} ---")
                avg_test_reward = self.test_policy(env, render=False, num_episodes=3)
                print(f"Reward medio nel test: {avg_test_reward:.2f}")
                print("--- Fine Test ---\n")
        
        print("Addestramento completato!")
        self.plot_training_stats()
    
    def test_policy(self, env, render=False, num_episodes=1, max_steps=1000):
        """
        Testa la policy appresa
        
        Args:
            env: Ambiente Gymnasium
            render: Se mostrare il rendering
            num_episodes: Numero di episodi di test
            max_steps: Numero massimo di passi per episodio
            
        Returns:
            float: Reward medio
        """
        total_rewards = []
        
        # Crea un ambiente separato per il rendering se necessario
        if render:
            test_env = gym.make('LunarLander-v2', render_mode='human')
        else:
            test_env = env
        
        for episode in range(num_episodes):
            if render:
                state, _ = test_env.reset()
            else:
                state, _ = env.reset()
            # Usa direttamente lo stato continuo
            
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
                # Usa direttamente lo stato continuo
                
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
        if not render:  # Stampa solo se non stiamo facendo rendering
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
        """Salva la policy appresa"""
        np.save(filename, dict(self.weights))
        print(f"Policy salvata in {filename}")
    
    def load_policy(self, filename):
        """Carica una policy salvata"""
        loaded_weights = np.load(filename, allow_pickle=True).item()
        self.weights = defaultdict(float, loaded_weights)
        print(f"Policy caricata da {filename}")