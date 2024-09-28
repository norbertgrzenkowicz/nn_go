package main

import (
	"errors"
	"fmt"
	"math"
)

// type perceptron struct {
// 	weights []float64
// 	bias float64
// 	output  float64
// }

func (p *perceptron) rELU(x float64) float64 {
	return math.Max(0, x)
}

func (p *perceptron) calc_output(input []float64) error {
	var sum float64

	if len(input) != len(p.weights) {
		return errors.New("weights and input of different lengths: " + fmt.Sprintf("%d vs %d", len(input), len(p.weights)))
	}

	for i, inp := range input {
		sum += inp * p.weights[i]
	}
	p.output = p.rELU(sum)
	return nil
}
