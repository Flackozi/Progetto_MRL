import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import gymnasium as gym

class LunarLanderClass:
    def __init__(self, numEpisodes, Alpha, initialEpsilon, Lambda, Gamma, k):
        """
        Inizializza la classe per l'apprendimento SARSA(λ) su Lunar Lander
        
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
        self.initialEpsilon = initialEpsilon
        self.lambda_param = Lambda
        self.gamma = Gamma
        self.k = k
        
        # Spazio delle azioni per Lunar Lander (4 azioni discrete)
        self.num_actions = 4
        
        # Parametri per la discretizzazione dello spazio degli stati
        self.state_bins = {
            'x': np.linspace(-1.5, 1.5, 5),          # posizione x
            'y': np.linspace(0, 1.5, 5),             # posizione y  
            'vx': np.linspace(-2, 2, 5),             # velocità x
            'vy': np.linspace(-2, 2, 5),             # velocità y
            'angle': np.linspace(-np.pi, np.pi, 5),  # angolo
            'angular_vel': np.linspace(-5, 5, 5),    # velocità angolare
            'leg1': [0, 1],                           # contatto gamba 1 (binario)
            'leg2': [0, 1]                            # contatto gamba 2 (binario)
        }
        
        # Pesi e eligibility traces
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
        print(f"Inizializzato stage {stage}")
        
    def discretize_state(self, state):
        """
        Discretizza lo stato continuo in uno stato discreto
        
        Args:
            state: Stato continuo dall'ambiente (8-dimensional array)
            
        Returns:
            tuple: Stato discretizzato
        """
        x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact = state
        
        # Discretizza ogni componente dello stato
        x_bin = np.digitize(x, self.state_bins['x']) - 1
        y_bin = np.digitize(y, self.state_bins['y']) - 1
        vx_bin = np.digitize(vx, self.state_bins['vx']) - 1
        vy_bin = np.digitize(vy, self.state_bins['vy']) - 1
        angle_bin = np.digitize(angle, self.state_bins['angle']) - 1
        angular_vel_bin = np.digitize(angular_vel, self.state_bins['angular_vel']) - 1
        
        # Assicura che gli indici siano nei limiti corretti
        x_bin = max(0, min(x_bin, len(self.state_bins['x']) - 1))
        y_bin = max(0, min(y_bin, len(self.state_bins['y']) - 1))
        vx_bin = max(0, min(vx_bin, len(self.state_bins['vx']) - 1))
        vy_bin = max(0, min(vy_bin, len(self.state_bins['vy']) - 1))
        angle_bin = max(0, min(angle_bin, len(self.state_bins['angle']) - 1))
        angular_vel_bin = max(0, min(angular_vel_bin, len(self.state_bins['angular_vel']) - 1))
        
        return (x_bin, y_bin, vx_bin, vy_bin, angle_bin, angular_vel_bin, 
                int(leg1_contact), int(leg2_contact))
    
    def get_state_action_key(self, state, action):
        """
        Crea una chiave unica per la coppia stato-azione
        
        Args:
            state: Stato discretizzato
            action: Azione
            
        Returns:
            tuple: Chiave per la coppia stato-azione
        """
        return (*state, action)
    
    def epsilon_greedy(self, state, episode):
        """
        Implementa la strategia epsilon-greedy per la selezione delle azioni
        
        Args:
            state: Stato discretizzato
            episode: Numero dell'episodio corrente
            
        Returns:
            int: Azione selezionata
        """
        # Calcola epsilon con decadimento
        epsilon = self.initialEpsilon / (1 + self.k * episode / self.numEpisodes)
        
        if np.random.random() < epsilon:
            # Azione casuale
            return np.random.randint(0, self.num_actions)
        else:
            # Azione greedy (migliore Q-value)
            q_values = []
            for action in range(self.num_actions):
                key = self.get_state_action_key(state, action)
                q_values.append(self.weights[key])
            
            # Restituisce l'azione con il Q-value più alto
            return np.argmax(q_values)
    
    def get_q_value(self, state, action):
        """
        Ottiene il Q-value per una coppia stato-azione
        
        Args:
            state: Stato discretizzato
            action: Azione
            
        Returns:
            float: Q-value
        """
        key = self.get_state_action_key(state, action)
        return self.weights[key]
    
    def update_eligibility_traces(self, state, action, decay=True):
        """
        Aggiorna le eligibility traces
        
        Args:
            state: Stato discretizzato
            action: Azione
            decay: Se applicare il decadimento alle traces esistenti
        """
        if decay:
            # Applica il decadimento a tutte le traces
            for key in list(self.eligibility_traces.keys()):
                self.eligibility_traces[key] *= self.gamma * self.lambda_param
                if abs(self.eligibility_traces[key]) < 1e-8:
                    del self.eligibility_traces[key]
        
        # Imposta la trace corrente a 1
        current_key = self.get_state_action_key(state, action)
        self.eligibility_traces[current_key] = 1.0
    
    def SARSALambda(self, env, render_episode_interval=None):
        """
        Implementa l'algoritmo SARSA(λ) per Lunar Lander
        
        Args:
            env: Ambiente Gymnasium
            render_episode_interval: Intervallo per il rendering (None = mai)
        """
        if not self.initialized:
            raise Exception("Devi chiamare initStage() prima di eseguire SARSA(λ)")
        
        print("Iniziando l'addestramento SARSA(λ)...")
        
        for episode in range(self.numEpisodes):
            # Reset dell'ambiente
            state, _ = env.reset()
            state = self.discretize_state(state)
            
            # Selezione azione iniziale
            action = self.epsilon_greedy(state, episode)
            
            episode_reward = 0
            episode_length = 0
            
            # Reset eligibility traces per ogni episodio
            self.eligibility_traces.clear()
            
            while True:
                # Esegui l'azione
                next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.discretize_state(next_state)
                
                episode_reward += reward
                episode_length += 1
                
                # Calcola Q(s,a)
                q_current = self.get_q_value(state, action)
                
                if terminated or truncated:
                    # Stato terminale: Q(s',a') = 0
                    td_error = reward - q_current
                    
                    # Aggiorna eligibility traces
                    self.update_eligibility_traces(state, action, decay=False)
                    
                    # Aggiorna tutti i pesi usando le eligibility traces
                    for key, trace in self.eligibility_traces.items():
                        self.weights[key] += self.alpha * td_error * trace
                    
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
                    for key, trace in self.eligibility_traces.items():
                        self.weights[key] += self.alpha * td_error * trace
                    
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
            
            # Test della policy senza rendering per evitare loop infiniti
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
            state = self.discretize_state(state)
            
            episode_reward = 0
            steps = 0  
            
            while steps < max_steps: 
                # Usa policy greedy (epsilon = 0)
                q_values = []
                for action in range(self.num_actions):
                    q_values.append(self.get_q_value(state, action))
                action = np.argmax(q_values)
                
                if render:
                    next_state, reward, terminated, truncated, _ = test_env.step(action)
                else:
                    next_state, reward, terminated, truncated, _ = env.step(action)
                next_state = self.discretize_state(next_state)
                
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
        ax1.set_title('Reward per Episodio')
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
