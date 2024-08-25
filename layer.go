package main

import (
	"log"
	"math"
)

type layer struct {
	perceptrons []perceptron
}

func (layer *layer) setWeightToLayer(weight float64) {
	// TODO: let perceptron have a pointer to the weights,
	// so we wont have to init whole new slice of weights

	for _, perceptron := range layer.perceptrons {
		for i := range perceptron.weights {
			perceptron.weights[i] = weight
		}
	}
}

func softmax(outputs *layer) []float64 {
	var sum float64
	var dupa []float64
	for _, p := range outputs.perceptrons {
		sum += math.Exp(p.output)
	}
	log.Println(sum)
	for _, output := range outputs.perceptrons {
		log.Println(output.output)
		dupa = append(dupa, math.Exp(output.output)/sum)
	}
	return dupa
}

func (l *layer) calc_output_in_layer(input []float64) ([]float64, error) {
	var outputs []float64
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

func (l *layer) calc_output_in_first_layer(input float64) ([]float64, error) {
	var outputs []float64
	for i, perc := range l.perceptrons {
		perc.output = input
		outputs = append(outputs, perc.output)
		l.perceptrons[i] = perc
	}

	return outputs, nil
}
