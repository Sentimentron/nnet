package mlp

import (
	"math"
)

type SoftmaxLayer struct {
	HiddenLayer
}

func NewSoftmaxLayer(numInputUnits, numHiddenUnits int, f LayerFunc, d LayerFunc) *SoftmaxLayer {
	h := new(SoftmaxLayer)
	h.NumInputUnits = numInputUnits
	h.NumHiddenUnits = numHiddenUnits
	return h
}

func (s *SoftmaxLayer) Forward(input []float64) []float64 {
	numOutputUnits := len(s.B)
	predicted := make([]float64, numOutputUnits)
	for i := range predicted {
		sum := 0.0
		for j := range input {
			sum += math.Exp(input[j])
		}
		predicted[i] = math.Exp(input[i]) / sum
	}
	return predicted
}

func (s *SoftmaxLayer) BackwardWithTarget(predicted, target []float64) []float64 {
	delta := make([]float64, s.NumHiddenUnits)
	for i := range delta {
		delta[i] = (predicted[i] - target[i])
	}
	return delta
}

func (s *SoftmaxLayer) Backward(predicted, accumulateDelta []float64) []float64 {
	delta := make([]float64, s.NumHiddenUnits)
	for i := range delta {
		delta[i] = accumulateDelta[i]
	}
	return delta
}

func (s *SoftmaxLayer) Turn(batch, deltas [][]float64, option TrainingOption) {
	// Do nothing: I am weightless
}
