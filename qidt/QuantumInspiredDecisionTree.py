import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class QuantumInspiredDecisionTree:
    def __init__(self, max_depth=3):
        self.max_depth = max_depth

    def _quantum_split(self, X, y):
        # Quantum-inspired probabilistic split
        probabilities = np.abs(np.fft.fft(X))  # Using Fourier Transform as a quantum-inspired operation
        split_point = np.argmax(probabilities)
        return split_point

    def fit(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            self.label = np.bincount(y).argmax()
            return

        split_point = self._quantum_split(X, y)
        self.split_point = split_point

        left_indices = X <= X[split_point]
        right_indices = X > X[split_point]

        self.left = QuantumInspiredDecisionTree(self.max_depth)
        self.right = QuantumInspiredDecisionTree(self.max_depth)

        self.left.fit(X[left_indices], y[left_indices], depth + 1)
        self.right.fit(X[right_indices], y[right_indices], depth + 1)

    def predict(self, X):
        if hasattr(self, 'label'):
            return self.label
        if X <= self.split_point:
            return self.left.predict(X)
        else:
            return self.right.predict(X)

# Load dataset
data = load_iris()
X = data.data[:, 0]  # Using only the first feature for simplicity
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
qidt = QuantumInspiredDecisionTree(max_depth=3)
qidt.fit(X_train, y_train)

# Predict
y_pred = np.array([qidt.predict(x) for x in X_test])

# Evaluate
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
