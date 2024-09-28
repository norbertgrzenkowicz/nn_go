package main

type layer struct {
	perceptrons []perceptron
	inputs      []float64
	layer_index int
}

func (layer *layer) setWeightToLayer() {
	for _, perceptron := range layer.perceptrons {
		for i := range perceptron.weights {
			perceptron.weights[i] = generateWeight(1.0, 4.0)
		}
	}
}

func (l *layer) calc_output_in_layer() ([]float64, error) {
	var outputs []float64

	if l.layer_index == 0 {
		for i, inp := range l.inputs {
			dupa := l.perceptrons[i].rELU(inp * l.perceptrons[i].weights[0])
			outputs = append(outputs, dupa)
			l.perceptrons[i].output = dupa
		}
		return outputs, nil
	}

	for i, perc := range l.perceptrons {
		err := perc.calc_output(l.inputs)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, perc.output)
		l.perceptrons[i] = perc
	}
	return outputs, nil
}
