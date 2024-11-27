import numpy as np

class NeuralNetwork:
    def __init__(self, input_size=784, hidden_size=128, output_size=10, learning_rate=0.1):
        # Initialize weights and biases
        self.weights1 = np.random.randn(input_size, hidden_size) * 0.01
        self.weights2 = np.random.randn(hidden_size, output_size) * 0.01
        self.bias1 = np.zeros((1, hidden_size))
        self.bias2 = np.zeros((1, output_size))
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        # Forward propagation
        self.hidden = self.sigmoid(np.dot(X, self.weights1) + self.bias1)
        self.output = self.softmax(np.dot(self.hidden, self.weights2) + self.bias2)
        return self.output

    def backward(self, X, y, output):
        # Backward propagation
        self.output_error = output - y
        self.output_delta = self.output_error

        self.hidden_error = np.dot(self.output_delta, self.weights2.T)
        self.hidden_delta = self.hidden_error * self.sigmoid_derivative(self.hidden)

        # Update weights and biases
        self.weights2 -= self.learning_rate * np.dot(self.hidden.T, self.output_delta)
        self.bias2 -= self.learning_rate * np.sum(self.output_delta, axis=0, keepdims=True)
        self.weights1 -= self.learning_rate * np.dot(X.T, self.hidden_delta)
        self.bias1 -= self.learning_rate * np.sum(self.hidden_delta, axis=0, keepdims=True)

    def train(self, X, y, epochs):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-8))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')

    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)

# Load and preprocess MNIST data
def load_mnist():
    # Load training data from file
    # Note: You'll need to download the MNIST dataset and specify the correct path
    with np.load('mnist.npz') as data:
        training_images = data['x_train']
        training_labels = data['y_train']
        test_images = data['x_test']
        test_labels = data['y_test']

    # Normalize and reshape images
    training_images = training_images.reshape(60000, 784).astype('float32') / 255
    test_images = test_images.reshape(10000, 784).astype('float32') / 255

    # Convert labels to one-hot encoding
    def to_one_hot(y, num_classes=10):
        return np.eye(num_classes)[y]

    training_labels = to_one_hot(training_labels)
    test_labels = to_one_hot(test_labels)

    return training_images, training_labels, test_images, test_labels

# Example usage:
if __name__ == "__main__":
    # Load MNIST data
    X_train, y_train, X_test, y_test = load_mnist()

    # Create and train neural network
    nn = NeuralNetwork()
    nn.train(X_train, y_train, epochs=1000)

    # Test the model
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == np.argmax(y_test, axis=1))
    print(f'Test accuracy: {accuracy:.4f}')
