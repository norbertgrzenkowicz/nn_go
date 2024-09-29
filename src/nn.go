package main

import (
	"fmt"
	"golang/mnist"
	"math"
	"math/rand"
)

const (
	learningRate = 0.001
	epsilon      = 1e-8
)

type neuralNetwork struct {
	layers []layer
}

func (nn *neuralNetwork) initNetwork(layerConfigs []struct {
	numPerceptrons int
	numWeights     int
	numInputs      int
}) error {
	nn.layers = make([]layer, len(layerConfigs))

	for i, config := range layerConfigs {
		nn.layers[i] = layer{
			perceptrons: make([]perceptron, config.numPerceptrons),
			inputs:      make([]float64, config.numInputs),
			layerIndex:  i,
		}

		for j := range nn.layers[i].perceptrons {
			nn.layers[i].perceptrons[j] = perceptron{
				weights: make([]float64, config.numWeights),
				bias:    rand.NormFloat64() * 0.1,
			}
			for k := range nn.layers[i].perceptrons[j].weights {
				nn.layers[i].perceptrons[j].weights[k] = rand.NormFloat64() * math.Sqrt(2.0/float64(config.numInputs))
			}
		}
	}

	return nil
}

func (nn *neuralNetwork) forward(input []float64) []float64 {
	nn.layers[0].inputs = input
	for i, layer := range nn.layers {
		outputs := make([]float64, len(layer.perceptrons))
		for j, perceptron := range layer.perceptrons {
			sum := perceptron.bias
			for k, weight := range perceptron.weights {
				sum += layer.inputs[k] * weight
			}
			if i == len(nn.layers)-1 {
				outputs[j] = sum
			} else {
				outputs[j] = relu(sum)
			}
			perceptron.output = outputs[j]
		}

		if i < len(nn.layers)-1 {
			nn.layers[i+1].inputs = outputs
		} else {
			return softmax(outputs)
		}
	}
	return nil
}

func (nn *neuralNetwork) backpropagation(input []float64, target int) {
	output := nn.forward(input)
	deltas := make([][]float64, len(nn.layers))
	for i := len(nn.layers) - 1; i >= 0; i-- {
		deltas[i] = make([]float64, len(nn.layers[i].perceptrons))
		if i == len(nn.layers)-1 {
			for j := range deltas[i] {
				if j == target {
					deltas[i][j] = output[j] - 1
				} else {
					deltas[i][j] = output[j]
				}
			}
		} else {
			for j := range deltas[i] {
				sum := 0.0
				for k, nextPerceptron := range nn.layers[i+1].perceptrons {
					sum += nextPerceptron.weights[j] * deltas[i+1][k]
				}
				deltas[i][j] = sum * reluDerivative(nn.layers[i].perceptrons[j].output)
			}
		}

		for j, perceptron := range nn.layers[i].perceptrons {
			for k := range perceptron.weights {
				perceptron.weights[k] -= learningRate * deltas[i][j] * nn.layers[i].inputs[k]
			}
			perceptron.bias -= learningRate * deltas[i][j]
		}
	}
}

func (nn *neuralNetwork) train(trainSet *mnist.Set, batchSize, epochs int) {
	for epoch := 0; epoch < epochs; epoch++ {
		totalLoss := 0.0
		for i := 0; i < len(trainSet.Images); i += batchSize {
			end := i + batchSize
			if end > len(trainSet.Images) {
				end = len(trainSet.Images)
			}

			batchLoss := 0.0
			for j := i; j < end; j++ {
				input := make([]float64, len(trainSet.Images[j]))
				for k, pixel := range trainSet.Images[j] {
					input[k] = float64(pixel) / 255.0
				}

				output := nn.forward(input)
				nn.backpropagation(input, int(trainSet.Labels[j]))

				batchLoss += crossEntropyLoss(output, int(trainSet.Labels[j]))
			}
			totalLoss += batchLoss
		}
		fmt.Printf("Epoch %d, Average Loss: %.4f\n", epoch+1, totalLoss/float64(len(trainSet.Images)))
	}
}

func (nn *neuralNetwork) predict(input []float64) int {
	output := nn.forward(input)
	maxIndex := 0
	maxValue := output[0]
	for i, value := range output {
		if value > maxValue {
			maxValue = value
			maxIndex = i
		}
	}
	return maxIndex
}

func (nn *neuralNetwork) evaluate(testSet *mnist.Set) float64 {
	correct := 0
	for i := range testSet.Images {
		input := make([]float64, len(testSet.Images[i]))
		for j, pixel := range testSet.Images[i] {
			input[j] = float64(pixel) / 255.0
		}
		prediction := nn.predict(input)
		if prediction == int(testSet.Labels[i]) {
			correct++
		}
	}
	return float64(correct) / float64(len(testSet.Images))
}
