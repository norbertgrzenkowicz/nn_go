package main

import (
	"fmt"
	"golang/mnist"
	"log"
)

var LEARNING_RATE = 0.1

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
			layer_index: i,
		}

		for j := range nn.layers[i].perceptrons {
			nn.layers[i].perceptrons[j] = perceptron{
				weights: make([]float64, config.numWeights),
			}
		}
		nn.layers[i].setWeightToLayer()
	}

	return nil
}

func (nn *neuralNetwork) train(input []float64, target float64) {

	nn.forward(input)
	nn.backpropagation(softmax(&nn.layers[3]), onehotlabels(int8(target)))
}

func (nn *neuralNetwork) forward(input []float64) {
	for _, layer := range nn.layers {
		output, err := layer.calc_output_in_layer(input)
		if err != nil {
			log.Fatal(err)
		}
		input = output
	}
}

func (nn *neuralNetwork) backpropagation(soft_max_output []float64, y_true []float64) {
	grad_loss := calc_initial_grad_loss(soft_max_output, y_true)

	for i := len(nn.layers) - 1; i >= 0; i-- {
		fmt.Print(len(nn.layers[3].perceptrons[0].weights), len(grad_loss))
		nn.update_weights(grad_loss)

		if i > 0 {
			grad_loss = calc_grad_loss(nn.layers[i], grad_loss)
		}
	}
}

func (nn *neuralNetwork) update_weights(grad_loss []float64) {
	for i := range nn.layers {
		layer := &nn.layers[i]
		inputs := layer.inputs // Use the stored inputs

		for j := range layer.perceptrons {
			perceptron := &layer.perceptrons[j]
			for k := range perceptron.weights {
				perceptron.weights[k] -= LEARNING_RATE * grad_loss[j] * inputs[k]
			}
		}
	}
}

func main() {
	train_set, test_set, err := mnist.Load("/home/norbert/repos/golang/")

	if err != nil {
		log.Fatal(err)
	}
	log.Println("Loaded", len(train_set.Images), "images")
	log.Println("Loaded", len(test_set.Images), "images")

	layerConfigs := []struct {
		numPerceptrons int
		numWeights     int
		numInputs      int
	}{
		{784, 1, 784},
		{16, 784, 784},
		{16, 16, 16},
		{10, 16, 16},
	}

	var nn neuralNetwork
	errr := nn.initNetwork(layerConfigs)

	if errr != nil {
		log.Fatal(errr)
	}

	var pixels []float64
	for _, pixel := range train_set.Images[0] {
		pixels = append(pixels, float64(pixel)/255.0)
	}

	nn.train(pixels, float64(train_set.Labels[0]))

}
