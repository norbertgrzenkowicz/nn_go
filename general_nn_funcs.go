package main

import (
	"math"
)

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
