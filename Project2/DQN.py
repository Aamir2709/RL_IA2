import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
from collections import deque
from tensorflow.keras.models import clone_model
import random
import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.preprocessing')


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)  # Replay memory
        self.gamma = 0.95  # Discount factor for future rewards
        self.epsilon = 1.0  # Exploration-exploitation trade-off
        self.epsilon_decay = 0.995  # Epsilon decay rate
        self.epsilon_min = 0.01  # Minimum epsilon value
        self.learning_rate = 0.0001  # Learning rate for the neural network
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Dense(128, input_dim=self.state_size, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=self.learning_rate))
        return model


    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        return np.argmax(self.model.predict(state)[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def preprocess_data(data):
    # Extract features (X) and target variable (y)
    features = data.drop(columns=['Price', 'ProductID'])  # Exclude 'ProductID' from features
    target = data['Price']

    # Define numerical and categorical columns
    numerical_cols = ['CompetitorPrice', 'HistoricalPrice']
    categorical_cols = ['Demand', 'CustomerSatisfaction', 'TimeOfDay', 'DayOfWeek']

    # Create a column transformer for efficient preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_cols),
            ('cat', OneHotEncoder(sparse=False, drop='first'), categorical_cols)
        ])

    X = preprocessor.fit_transform(features)
    y = target.values
    return X, y

def create_replay_memory(X_train, y_train, agent, batch_size):
    memory = deque(maxlen=2000)
    for _ in range(len(X_train)):
        state = X_train[_].reshape(1, -1)
        action = agent.act(state)
        reward = -np.abs(y_train[_] - action)
        next_state = X_train[random.randint(0, len(X_train) - 1)].reshape(1, -1)
        done = False
        memory.append((state, action, reward, next_state, done))
    return memory

def train():
    # Load the CSV file into a Pandas DataFrame
    data = pd.read_csv('pricing_data.csv')

    # Preprocess the data
    X, y = preprocess_data(data)

    # Suppress specific warnings
    warnings.filterwarnings(action='ignore', category=FutureWarning, module='sklearn.preprocessing')

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create an instance of the DQNAgent class
    agent = DQNAgent(state_size=X_train.shape[1], action_size=1)  # Assuming 1 action (price adjustment)

    # Define parameters for training
    batch_size = 32
    epochs = 200  # Increased epochs for more training iterations

    # Create replay memory buffer
    memory = create_replay_memory(X_train, y_train, agent, batch_size)

    for epoch in range(epochs):
        minibatch = random.sample(memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + agent.gamma * np.amax(agent.model.predict(next_state)[0])
            target_f = agent.model.predict(state)
            target_f[0][action] = target
            agent.model.fit(state, target_f, epochs=1, verbose=0)

        # Optionally, evaluate the agent on the test data after each epoch
        test_loss = agent.model.evaluate(X_test, y_test, verbose=0)
        print(f'Epoch {epoch + 1}/{epochs}, Test Loss: {test_loss:.4f}')

# Call the train function to initiate training
train()