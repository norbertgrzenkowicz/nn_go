package main

import (
	"fmt"
	"golang/mnist"
	"log"
	"os"
)

func main() {
	dir, err := os.Getwd()
	if err != nil {
		fmt.Println("Error:", err)
	}

	trainSet, testSet, err := mnist.Load(dir)
	if err != nil {
		log.Fatal(err)
	}
	log.Println("Loaded", len(trainSet.Images), "training images")
	log.Println("Loaded", len(testSet.Images), "test images")

	layerConfigs := []struct {
		numPerceptrons int
		numWeights     int
		numInputs      int
	}{
		{784, 1, 784},
		{128, 784, 784},
		{32, 128, 128},
		{10, 32, 32},
	}

	var nn neuralNetwork
	err = nn.initNetwork(layerConfigs)
	if err != nil {
		log.Fatal(err)
	}

	// Train the network
	nn.train(trainSet, 128, 10) // Batch size of 32, 10 epochs

	// Evaluate the network
	accuracy := nn.evaluate(testSet)
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)

	// Example prediction
	testImage := make([]float64, len(testSet.Images[0]))
	for i, pixel := range testSet.Images[0] {
		testImage[i] = float64(pixel) / 255.0
	}
	prediction := nn.predict(testImage)
	fmt.Printf("Prediction for first test image: %d (actual: %d)\n", prediction, testSet.Labels[0])
}
