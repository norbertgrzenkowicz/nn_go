package main

import (
	"math"
)

type layer struct {
	perceptrons []perceptron
	inputs      []float64
	layer_index int
}

func (layer *layer) setWeightToLayer() {
	// TODO: let perceptron have a pointer to the weights,
	// so we wont have to init whole new slice of weights

	for _, perceptron := range layer.perceptrons {
		for i := range perceptron.weights {
			perceptron.weights[i] = generateWeight(1.0, 4.0)
		}
	}
}

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

func (l *layer) calc_output_in_layer(input []float64) ([]float64, error) {
	var outputs []float64

	if l.layer_index == 0 {
		for i, inp := range input {
			dupa := l.perceptrons[i].rELU(inp * l.perceptrons[i].weights[0])
			outputs = append(outputs, dupa)
			l.perceptrons[i].output = dupa
		}

		return outputs, nil
	}

	for i, perc := range l.perceptrons {
		err := perc.calc_output(input)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, perc.output)
		l.perceptrons[i] = perc
	}
	return outputs, nil
}
