### Neural Network from Scratch in Go

---

# Neural Network in Go

This project implements a neural network for image classification in MNIST dataset. 
The network uses multiple layers of perceptrons, ReLU activation, and softmax for the output layer to classify handwritten digits.

## Features
- Neural network with configurable layers and perceptrons.
- Utilizes ReLU activation function.
- Softmax output layer for classification.
- Training and evaluation on the MNIST dataset.
- Current test accuracy: ~30% (1st smoke tests, please be patient..)
- Configurable network from file/args will be done when I'll get bored.

## Prerequisites
- Go 1.16 or higher
- MNIST dataset
- Docker (for running via Docker)

## File Structure
- **`nn.go`**: Main file to initialize, train, and evaluate the neural network.
- **`layer.go`**: Defines the layer  structures along with forward propagation and weight updates.
- **`general_nn_funcs.go`**: Utility functions for weight initialization and gradient loss calculation.
- **`perceptron.go`**: Defines perceptron structure.

## How to Run Locally

1. Clone the repository.

2. Install dependencies:
    ```bash
    go get -u github.com/petar/GoMNIST
    ```

3. Run the training and evaluation:
    ```bash
    go run nn.go
    ```
4. The accuracy will be displayed in the terminal after evaluation.

## Running with Docker

1. Build the Docker image:
    ```bash
    docker build -t nn_go .
    ```

2. Run the container:
    ```bash
    docker run -it nn_go
    ```

3. The output will display the training process and final accuracy.


