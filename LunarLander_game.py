import gymnasium as gym
import lunarLanderClasses as LLC

#--------------------------------------------
#Studenti: Flavio Campobasso 0357289, Carlo Maria Fioramanti 0357331
#--------------------------------------------
#Lunar Lander è un ambiente di simulazione che rappresenta un problema 
#classico di ottimizzazione della traiettoria di un razzo. 
#L'obiettivo del gioco è controllare un modulo lunare e farlo atterrare 
# in sicurezza su una piattaforma di atterraggio situata all'interno di un ambiente simulato. 
#Il gioco richiede la gestione delle forze di gravità e l'uso dei motori del razzo 
#per regolare l'orientamento e la velocità, al fine di prevenire un atterraggio accidentale.
#--------------------------------------------


#con render_mode = "human" possiamo rappresentare graficamente il gioco

# Crea l'ambiente Lunar Lander
env = gym.make('LunarLander-v3')

# Numero di episodi e parametri di apprendimento
numEpisodes = 4000
Alpha = 0.15
initialEpsilon = 0.2
Lambda = 0.8
Gamma = 0.99
k = 0.8

# Classe per l'apprendimento (definita nel file lunarLanderClasses.py)
Class = LLC.LunarLanderClass(numEpisodes, Alpha, initialEpsilon, Lambda, Gamma, k)
Class.initStage(1)

# Eseguo l'algoritmo SARSA(λ) su Lunar Lander
Class.SARSALambda(env, 500)

env.close()
