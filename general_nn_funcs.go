package main

import (
	"log"
	"math"
	"math/rand"
)

func softmax(outputs *layer) []float64 {
	var max float64 = math.Inf(-1)
	for _, output := range outputs.perceptrons {
		if output.output > max {
			max = output.output
		}
	}

	var sum float64
	for _, output := range outputs.perceptrons {
		sum += math.Exp(output.output - max)
	}

	var result []float64
	for _, output := range outputs.perceptrons {
		result = append(result, math.Exp(output.output-max)/sum)
	}

	return result
}

func calc_initial_grad_loss(soft_max_output, y_true []float64) []float64 {
	grad_loss := make([]float64, len(soft_max_output))
	for i := range soft_max_output {
		grad_loss[i] = soft_max_output[i] - y_true[i]
	}
	return grad_loss
}

func calc_grad_loss(layer layer, next_grad_loss []float64) []float64 {
	// Initialize gradient loss for this layer
	grad_loss := make([]float64, len(layer.perceptrons))

	for j := range layer.perceptrons { // Iterate over each neuron in the hidden layer
		sum := 0.0
		for k := range next_grad_loss { // Iterate over each neuron in the next layer (output layer)
			weight := layer.perceptrons[j].weights[k] // weight from hidden neuron `j` to output neuron `k`
			sum += next_grad_loss[k] * weight
		}
		// Calculate the final gradient for hidden neuron `j`
		grad_loss[j] = sum * layer.perceptrons[j].output
	}

	return grad_loss
}

func calc_grad_loss_05(layer layer, next_grad_loss []float64) []float64 {
	grad_loss := make([]float64, len(layer.perceptrons))
	for i, neuron := range layer.perceptrons {
		for j, weight := range neuron.weights {
			grad_loss[i] += next_grad_loss[j] * weight * neuron.output
		}
	}
	return grad_loss
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

func generateWeight(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
