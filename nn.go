package main

import (
	"fmt"
	"golang/mnist"
	"log"
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
type perceptron struct {
	weights []float64
	bias    float64
	output  float64
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
			layer_index: i,
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

func relu(x float64) float64 {
	return math.Max(0, x)
}

func reluDerivative(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}

func softmax(x []float64) []float64 {
	max := x[0]
	for _, v := range x[1:] {
		if v > max {
			max = v
		}
	}

	exp := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		exp[i] = math.Exp(v - max)
		sum += exp[i]
	}

	for i := range exp {
		exp[i] /= sum
	}
	return exp
}

func crossEntropyLoss(predictions []float64, target int) float64 {
	return -math.Log(predictions[target] + epsilon)
}

func main() {
	train_set, test_set, err := mnist.Load("/home/norbert/repos/golang/")
	if err != nil {
		log.Fatal(err)
	}
	log.Println("Loaded", len(train_set.Images), "training images")
	log.Println("Loaded", len(test_set.Images), "test images")

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
	nn.train(train_set, 128, 10) // Batch size of 32, 10 epochs

	// Evaluate the network
	accuracy := nn.evaluate(test_set)
	fmt.Printf("Test Accuracy: %.2f%%\n", accuracy*100)

	// Example prediction
	testImage := make([]float64, len(test_set.Images[0]))
	for i, pixel := range test_set.Images[0] {
		testImage[i] = float64(pixel) / 255.0
	}
	prediction := nn.predict(testImage)
	fmt.Printf("Prediction for first test image: %d (actual: %d)\n", prediction, test_set.Labels[0])
}
