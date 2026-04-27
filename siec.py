import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split


class SimpleNeuralNetwork:
    def __init__(self, hidden_size=100, learning_rate=0.01, epochs=1000, random_state=1):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.random_state = random_state

        self.W1 = None
        self.B1 = None
        self.W2 = None
        self.B2 = None

        self.mean_ = None
        self.std_ = None

    def relu(self, X):
        return np.maximum(X, 0)

    def relu_derivative(self, X):
        return np.where(X > 0, 1, 0)

    def softmax(self, X):
        # Stabilizacja numeryczna: odejmujemy maksimum przed exp().
        X_shifted = X - np.max(X, axis=1, keepdims=True)
        exp_X = np.exp(X_shifted)
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)

    def one_hot(self, y, number_of_classes):
        y_one_hot = np.zeros((y.shape[0], number_of_classes))
        y_one_hot[np.arange(y.shape[0]), y] = 1
        return y_one_hot

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)

        # Standaryzacja
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        X = (X - self.mean_) / self.std_

        number_of_samples = X.shape[0]
        number_of_features = X.shape[1]
        number_of_classes = len(np.unique(y))
        y_one_hot = self.one_hot(y, number_of_classes)

        # Architektura: 2 wejscia -> hidden_size neuronow -> 3 klasy.
        self.W1 = rgen.rand(number_of_features, self.hidden_size) - 0.5
        self.B1 = rgen.rand(1, self.hidden_size) - 0.5
        self.W2 = rgen.rand(self.hidden_size, number_of_classes) - 0.5
        self.B2 = rgen.rand(1, number_of_classes) - 0.5

        for epoch in range(1, self.epochs + 1):
            # Propagacja w przod.
            Z1 = X @ self.W1 + self.B1
            A1 = self.relu(Z1)
            Z2 = A1 @ self.W2 + self.B2
            A2 = self.softmax(Z2)

            # Propagacja wsteczna dla softmax + one-hot.
            dZ2 = A2 - y_one_hot
            dW2 = (A1.T @ dZ2) / number_of_samples
            dB2 = np.sum(dZ2, axis=0, keepdims=True) / number_of_samples

            dA1 = dZ2 @ self.W2.T
            dZ1 = dA1 * self.relu_derivative(Z1)
            dW1 = (X.T @ dZ1) / number_of_samples
            dB1 = np.sum(dZ1, axis=0, keepdims=True) / number_of_samples

            # Aktualizacja parametrow metoda spadku gradientu.
            self.W2 -= self.learning_rate * dW2
            self.B2 -= self.learning_rate * dB2
            self.W1 -= self.learning_rate * dW1
            self.B1 -= self.learning_rate * dB1

            if epoch % 100 == 0:
                loss = -np.mean(np.sum(y_one_hot * np.log(A2 + 1e-15), axis=1))
                print(f"Epoch: {epoch}, loss: {loss:.4f}")

        return self

    def predict_proba(self, X):
        X = (X - self.mean_) / self.std_
        Z1 = X @ self.W1 + self.B1
        A1 = self.relu(Z1)
        Z2 = A1 @ self.W2 + self.B2
        return self.softmax(Z2)

    def predict(self, X):
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)


def plot_decision_regions(X, y, classifier, resolution=0.02):
    x1_min, x1_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    x2_min, x2_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx1, xx2 = np.meshgrid(
        np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution)
    )

    grid_points = np.array([xx1.ravel(), xx2.ravel()]).T
    predicted_classes = classifier.predict(grid_points)
    predicted_classes = predicted_classes.reshape(xx1.shape)

    plt.contourf(xx1, xx2, predicted_classes, alpha=0.3)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="black")


def main():
    iris = datasets.load_iris()
    X = iris.data[:, [2, 3]]
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.3,
        random_state=1,
        stratify=y
    )

    model = SimpleNeuralNetwork(
        hidden_size=100,
        learning_rate=0.01,
        epochs=1000,
        random_state=1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)

    print("\nPredicted classes:")
    print(y_pred)

    print("\nTrue classes:")
    print(y_test)

    print("\nAccuracy:")
    print(accuracy)

    proba = model.predict_proba(X_test[:5])
    predictions = model.predict(X_test[:5])

    print("\nClass probabilities for the first 5 test samples:")
    print("sample | setosa | versicolor | virginica | predicted class")
    print("-" * 62)

    for i, row in enumerate(proba):
        print(
            f"{i + 1:>6} | "
            f"{row[0]:>6.4f} | "
            f"{row[1]:>10.4f} | "
            f"{row[2]:>9.4f} | "
            f"{predictions[i]}"
        )

    plot_decision_regions(X_train, y_train, classifier=model)
    plt.xlabel("petal length [cm]")
    plt.ylabel("petal width [cm]")
    plt.title("Simple neural network - Iris classification")
    plt.show()


if __name__ == "__main__":
    main()
