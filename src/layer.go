package main

type layer struct {
	perceptrons []perceptron
	inputs      []float64
	layerIndex  int
}

func (l *layer) CalcOutputInLayer() ([]float64, error) {
	var outputs []float64

	if l.layerIndex == 0 {
		for i, inp := range l.inputs {
			dupa := l.perceptrons[i].rELU(inp * l.perceptrons[i].weights[0])
			outputs = append(outputs, dupa)
			l.perceptrons[i].output = dupa
		}
		return outputs, nil
	}

	for i, perc := range l.perceptrons {
		err := perc.calcOutput(l.inputs)
		if err != nil {
			return nil, err
		}
		outputs = append(outputs, perc.output)
		l.perceptrons[i] = perc
	}
	return outputs, nil
}
