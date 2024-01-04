import random

import string

TARGET = "SAHYADRI"

POPULATION_SIZE = 100

MUTATION_RATE = 0.2

def calculate_fitness(candidate):

 return sum(c != t for c, t in zip(candidate, TARGET))

def crossover(parent1, parent2): #cp--> Crossover Point

 cp = random.randint(1, len(TARGET) - 1)

 return parent1[:cp] + parent2[cp:]

def mutate(candidate): #mi --> Mutated Index

 mi = random.randint(0, len(candidate) - 1)

 return candidate[:mi] + random.choice(string.ascii_uppercase) + candidate[mi + 1:]

def main():

 population = ["XYZABHPQ"] # RAndom string. Length should be same as TARGET

 for generation in range(50):

 population = sorted(population, key=calculate_fitness)

 if calculate_fitness(population[0]) == 0:

 break

 new_generation = population[:10] # Elitism: Keep the top 10% of fittest individuals

 for _ in range(90):

 parent1, parent2 = random.choices(population[:50],k=2)

 child = crossover(parent1, parent2)

 if random.random() < MUTATION_RATE:

 child = mutate(child)

 new_generation.append(child)

 population = new_generation

 print(f"Generation: {generation + 1}\tString: {population[0]}\tFitness: 

{calculate_fitness(population[0])}")

 print(f"Generation: {generation + 1}\tString: {population[0]}\tFitness: 

{calculate_fitness(population[0])}")

if __name__ == "__main__":

 main()import random

import string

TARGET = "SAHYADRI"

POPULATION_SIZE = 100

MUTATION_RATE = 0.2

def calculate_fitness(candidate):

 return sum(c != t for c, t in zip(candidate, TARGET))

def crossover(parent1, parent2): #cp--> Crossover Point

 cp = random.randint(1, len(TARGET) - 1)

 return parent1[:cp] + parent2[cp:]

def mutate(candidate): #mi --> Mutated Index

 mi = random.randint(0, len(candidate) - 1)

 return candidate[:mi] + random.choice(string.ascii_uppercase) + candidate[mi + 1:]

def main():

 population = ["XYZABHPQ"] # RAndom string. Length should be same as TARGET

 for generation in range(50):

 population = sorted(population, key=calculate_fitness)

 if calculate_fitness(population[0]) == 0:

 break

 new_generation = population[:10] # Elitism: Keep the top 10% of fittest individuals

 for _ in range(90):

 parent1, parent2 = random.choices(population[:50],k=2)

 child = crossover(parent1, parent2)

 if random.random() < MUTATION_RATE:

 child = mutate(child)

 new_generation.append(child)

 population = new_generation

 print(f"Generation: {generation + 1}\tString: {population[0]}\tFitness: 

{calculate_fitness(population[0])}")

 print(f"Generation: {generation + 1}\tString: {population[0]}\tFitness: 

{calculate_fitness(population[0])}")

if __name__ == "__main__":

 main()


 2------------
 import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn import datasets
import matplotlib.pyplot as plt
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target
# K-means clustering
kmeans = KMeans(n_clusters=3)
kmeans_labels = kmeans.fit_predict(X)
# EM clustering
em = GaussianMixture(n_components=3)
em_labels = em.fit_predict(X)
# Visualize the results
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=kmeans_labels, cmap='plasma', edgecolor='k')
plt.title('K-means Clustering')
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=em_labels, cmap='plasma', edgecolor='k')
plt.title('EM Clustering')
plt.show()

3------
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
# Load the hotel bills vs tips dataset
tips = pd.read_csv('tips.csv')
# Extracting features (total bill) and target variable (tip)
X , y = tips['total_bill'] , tips['tip']
# Use lowess for Locally Weighted Regression
lowess = sm.nonparametric.lowess(y, X, frac=0.03)
# Plotting the results
plt.scatter(X, y, label='Original data')
plt.plot(lowess[:, 0], lowess[:, 1], color='red', label='Locally Weighted Regression')
plt.xlabel('Total Bill')
plt.ylabel('Tip')
plt.title('Locally Weighted Regression: Hotel Bills vs Tips')
plt.show()

4---------
import numpy as np

import tensorflow as tf

# Data normalization

X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float) / np.amax(np.array(([2, 9], [1, 5], [3, 6])), axis=0)

y = np.array(([92], [86], [89]), dtype=float) / 100

# Build the neural network model

model = tf.keras.Sequential([

 tf.keras.layers.Input(shape=(2,)),

 tf.keras.layers.Dense(300, activation='sigmoid'),

 tf.keras.layers.Dense(1, activation='sigmoid')

])

# Compile the model

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),

 loss='mean_squared_error')

# Training loop

for i in range(5):

 # Train the model for one epoch

 model.fit(X, y, epochs=1, verbose=0)

 

 # Predict and print results

 predictions = model.predict(X)

 print(f"-----------Epoch-{i+1} Starts----------")

 print("Input:\n", X)

 print("Actual Output:\n", y)

 print("Predicted Output:\n", predictions)

 print(f"-----------Epoch-{i+1} Ends----------\n")
 
6----------

import numpy as np

# Environment

num_states = 5

num_actions = 4

Q = np.zeros((num_states, num_actions))

# Q-learning parameters

alpha = 0.1 # learning rate

gamma = 0.9 # discount factor

epsilon = 0.1 # exploration-exploitation trade-off

# Q-learning training

for _ in range(1000):

 state = np.random.randint(0, num_states)

 if np.random.rand() < epsilon:

 action = np.random.randint(0, num_actions)

 else:

 action = np.argmax(Q[state, :])

 reward = np.random.normal(0, 1) # Simulated reward

 next_state = np.random.randint(0, num_states)

 Q[state, action] = (1 - alpha) * Q[state, action] + alpha * (reward + gamma * 

np.max(Q[next_state, :]))

# Q-values after training

print("Q-values:")

print(Q)
