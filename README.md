# Simple Neural Network - Iris Classification

This project implements a simple feedforward neural network from scratch using Python and NumPy. The model is trained on the classic Iris dataset and performs multi-class classification based on two flower features: petal length and petal width.

The goal of this project was to understand the internal mechanics of a neural network without relying on high-level machine learning frameworks such as TensorFlow or PyTorch. The implementation includes forward propagation, backpropagation, gradient descent, ReLU activation, softmax output, one-hot encoding, and decision region visualization.

## Project Overview

The neural network classifies Iris flowers into three classes:

- Setosa
- Versicolor
- Virginica

The model uses two input features from the Iris dataset:

- Petal length
- Petal width

The network architecture is intentionally simple:

```text
Input layer: 2 features
Hidden layer: 100 neurons, ReLU activation
Output layer: 3 neurons, softmax activation
```

## Features

- Custom neural network implementation using NumPy
- Manual data standardization
- One-hot encoding for multi-class labels
- ReLU activation function
- Softmax output layer
- Backpropagation implemented manually
- Gradient descent parameter updates
- Training loss printed during learning
- Accuracy evaluation on a test set
- Class probability output for selected samples
- Decision region visualization using Matplotlib

## Technologies Used

- Python
- NumPy
- Matplotlib
- scikit-learn

## Dataset

The project uses the Iris dataset available in `sklearn.datasets`. The dataset contains measurements of Iris flowers from three different species.

In this implementation, only two features are used:

```python
X = iris.data[:, [2, 3]]
```

This selects:

- Petal length
- Petal width

Using two features makes it possible to visualize the decision regions in a 2D plot.

## How the Model Works

The model is implemented in the `SimpleNeuralNetwork` class.

### 1. Data Standardization

Before training, the input data is standardized using the mean and standard deviation calculated from the training set:

```python
X = (X - self.mean_) / self.std_
```

The same stored mean and standard deviation are later used during prediction.

### 2. Forward Propagation

The network calculates hidden layer activations using ReLU and output probabilities using softmax:

```python
Z1 = X @ self.W1 + self.B1
A1 = self.relu(Z1)
Z2 = A1 @ self.W2 + self.B2
A2 = self.softmax(Z2)
```

### 3. Backpropagation

The gradients are calculated manually using the difference between predicted probabilities and one-hot encoded target labels:

```python
dZ2 = A2 - y_one_hot
```

The model then computes gradients for both layers and updates weights and biases using gradient descent.

### 4. Prediction

Predictions are made by selecting the class with the highest probability:

```python
return np.argmax(probabilities, axis=1)
```

## Example Output

During training, the model prints the loss every 100 epochs:

```text
Epoch: 100, loss: ...
Epoch: 200, loss: ...
...
Epoch: 1000, loss: ...
```

After training, the program displays:

- Predicted classes
- True classes
- Accuracy
- Class probabilities for the first 5 test samples
- Decision region plot

Example probability table:

```text
Class probabilities for the first 5 test samples:
sample | setosa | versicolor | virginica | predicted class
--------------------------------------------------------------
     1 | 0.0001 |     0.9123 |    0.0876 | 1
     2 | 0.9987 |     0.0012 |    0.0001 | 0
```

The exact values may differ depending on random initialization and training parameters.

## Decision Region Visualization

The project includes a custom `plot_decision_regions()` function that visualizes how the trained model separates the three Iris classes based on petal length and petal width.

```markdown
images/decision_regions.png
```

To add a screenshot to GitHub:

1. Create an `images` folder in the repository.
2. Save the plot screenshot as `decision_regions.png`.
3. Place it inside the `images` folder.
4. Keep the Markdown path as shown above.

## Project Structure

```text
.
├── siec.py
├── README.md
└── images/
    └── decision_regions.png
```

## How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
cd your-repository-name
```

### 2. Install dependencies

```bash
pip install numpy matplotlib scikit-learn
```

### 3. Run the script

```bash
python siec.py
```

## What I Learned

Through this project, I practiced and improved my understanding of:

- How a basic neural network is structured
- How forward propagation works
- How backpropagation updates model parameters
- Why softmax is used for multi-class classification
- How one-hot encoding represents target classes
- How feature standardization affects training
- How to evaluate classification accuracy
- How decision boundaries can be visualized in 2D

## Possible Improvements

Possible future improvements include:

- Adding a validation set
- Plotting the loss curve over epochs
- Making the number of hidden layers configurable
- Adding mini-batch gradient descent
- Comparing the custom implementation with `sklearn.neural_network.MLPClassifier`
- Saving trained model parameters to a file

## Author

Tomasz Murach
