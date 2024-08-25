package main

import (
	"golang/mnist"
	"log"
)

type neuralNetwork struct {
	layers []layer
}

func (nn *neuralNetwork) initNetwork(layerConfigs []struct {
	numPerceptrons int
	numWeights     int
	numInputs      int
},
	weight float64) error {
	nn.layers = make([]layer, len(layerConfigs))

	for i, config := range layerConfigs {
		nn.layers[i] = layer{
			perceptrons: make([]perceptron, config.numPerceptrons),
		}

		for j := range nn.layers[i].perceptrons {
			nn.layers[i].perceptrons[j] = perceptron{
				weights: make([]float64, config.numWeights),
			}
		}
		nn.layers[i].setWeightToLayer(weight)
	}

	return nil
}

func (nn *neuralNetwork) train(input []float64, target float64) {

	nn.forward(input)
}

func (nn *neuralNetwork) forward(input []float64) {

	var dupa []float64
	for _, layer := range nn.layers {
		if dupa != nil {
			input = dupa
		}
		output, err := layer.calc_output_in_layer(input)
		if err != nil {
			log.Fatal(err)
		}

		input = output
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
		{784, 784, 784},
		{16, 784, 784},
		{16, 16, 16},
		{10, 16, 16},
	}

	var nn neuralNetwork
	errr := nn.initNetwork(layerConfigs, 0.01)

	if errr != nil {
		log.Fatal(errr)
	}

	var pixels []float64
	for _, pixel := range train_set.Images[0] {
		pixels = append(pixels, float64(pixel)/255.0)
	}

	nn.train(pixels, float64(train_set.Labels[0]))

	dupa := softmax(&nn.layers[0])

	log.Println(dupa)
}
