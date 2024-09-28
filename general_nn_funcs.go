package main

import (
	"fmt"
	"math/rand"
)

// func softmax(outputs *layer) []float64 {
// 	var max float64 = math.Inf(-1)
// 	for _, output := range outputs.perceptrons {
// 		if output.output > max {
// 			max = output.output
// 		}
// 	}

// 	var sum float64
// 	for _, output := range outputs.perceptrons {
// 		sum += math.Exp(output.output - max)
// 	}

// 	var result []float64
// 	for _, output := range outputs.perceptrons {
// 		result = append(result, math.Exp(output.output-max)/sum)
// 	}

// 	return result
// }

func calc_initial_grad_loss(soft_max_output, y_true []float64) []float64 {
	grad_loss := make([]float64, len(soft_max_output))
	for i := range soft_max_output {
		grad_loss[i] = soft_max_output[i] - y_true[0]
	}
	fmt.Println("Init grad loss length ", len(grad_loss))
	return grad_loss
}

func calc_last_grad_loss(layer *layer, next_grad_loss []float64) []float64 {
	grad_loss := make([]float64, len(layer.perceptrons))
	for j := range layer.perceptrons {
		sum := 0.0
		for k := range next_grad_loss {
			sum += next_grad_loss[k] * layer.perceptrons[j].weights[0]
		}
		grad_loss[j] = sum * layer.perceptrons[j].output * (1 - layer.perceptrons[j].output)
	}
	return grad_loss
}

func calc_grad_loss(layer *layer, next_grad_loss []float64) []float64 {
	grad_loss := make([]float64, len(layer.perceptrons))
	for j := range grad_loss {
		sum := 0.0
		for k := 0; k < min(len(next_grad_loss), len(layer.perceptrons[j].weights)); k++ {
			sum += next_grad_loss[k] * layer.perceptrons[j].weights[k]
		}
		grad_loss[j] = sum * layer.perceptrons[j].output * (1 - layer.perceptrons[j].output)
	}
	return grad_loss
}

func generateWeight(min, max float64) float64 {
	return min + rand.Float64()*(max-min)
}
