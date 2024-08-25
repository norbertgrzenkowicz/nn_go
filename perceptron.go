package main

import (
	"errors"
	"fmt"
)

type perceptron struct {
	weights []float64
	output  float64
}

func (p *perceptron) rELU(x float64) float64 {
	if x > 0 {
		return x
	} else {
		return 0
	}
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

func (p *perceptron) predict() {
	// TODO
}

func (p *perceptron) test() {
	// TODO
}
