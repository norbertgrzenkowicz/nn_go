package main

import (
	"fmt"
	"golang/mnist"
	"log"
	"math"
	"math/rand"
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

func generateWeight(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}

func (nn *neuralNetwork) train(input []float64, target float64) {

	nn.forward(input)
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

func (nn *neuralNetwork) backpropagation(soft_max_output []float64) {
	for i := len(nn.layers) - 1; i >= 0; i-- {
		if i == len(nn.layers)-1 {
			nn.layers[i].update_weights(soft_max_output)
		} else {
			nn.layers[i].update_weights(grad_loss)
			// grad_loss :=  something to update grad_loss everytiem
		}
	}
}

func onehotlabels(label int8) []float64 {
	if label < 0 || label > 9 {
		log.Fatal("label must be between 0 and 9")
	}
	onehot := make([]float64, 10)
	onehot[label] = 1
	return onehot
}

func CE_loss(predictions []float64, targets []float64) float64 {
	epsilon := 1e-10 // Small constant to avoid log(0)
	loss := 0.0

	for i := range predictions {
		loss += targets[i] * math.Log(predictions[i]+epsilon)
	}

	return -loss
}

func calculate_grad_loss(predictions []float64, targets []float64) []float64 {

	grad_loss := make([]float64, len(predictions))
	for i := range predictions {
		grad_loss[i] = predictions[i] - targets[i]
	}
	return grad_loss
}

func (l *layer) update_weights(grad_loss []float64) {

	for i := range l.perceptrons {
		for j := range l.perceptrons[i].weights {
			l.perceptrons[i].weights[j] -= LEARNING_RATE * grad_loss[j]
		}
	}
}

// How to implement a simple neural network in Go
// https://appliedgo.net/perceptron/
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

	fmt.Println(CE_loss(softmax(&nn.layers[3]), onehotlabels(train_set.Labels[0])))
}
