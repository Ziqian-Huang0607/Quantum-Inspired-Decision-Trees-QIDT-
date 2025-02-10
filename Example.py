from qidt import QuantumInspiredDecisionTree
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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
