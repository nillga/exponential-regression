package exponential_regression

import (
	"errors"
	"github.com/sajari/regression"
	"math"
)

var (
	ErrAlreadyInitialized = errors.New("data has been initialized already")
	ErrAlreadyConverted = errors.New("data has been converted already")
	ErrNotEnoughData = errors.New("sample size is too small")
	ErrLinearRegression = errors.New("error at Linear Regression Level")
	ErrNotConverted = errors.New("unconverted data")
	ErrNegativeValue = errors.New("negative Value -> No exponential regression possible")
	ErrRanAlready = errors.New("regression was run already")
	ErrNotRan = errors.New("regression has not been run yet")
)

type Value struct {
	x float64
	y float64
}

type Input struct {
	values []Value
	covered []float64
	converted bool
}

type Output struct {
	a float64
	b float64
	converted bool
	err error
}

type Regression struct {
	input Input
	initialized bool
	ran bool
	output Output
}

func (r *Regression) Init (m map[float64]float64) error {
	if r.initialized {
		return ErrAlreadyInitialized
	}
	for x,y := range m {
		err := r.Append(x, y)
		if err != nil {
			return err
		}
	}
	r.initialized = true
	return nil
}

func (r *Regression) Append (x,y float64) error {
	if r.input.converted {
		return ErrAlreadyConverted
	}
	if y < 0 {
		return ErrNegativeValue
	}
	if r.input.hasAlready(x) {
		return nil
	}

	r.input.values = append(r.input.values, Value{x,y})
	r.input.covered = append(r.input.covered, x)
	r.initialized = true
	return nil
}

func (i *Input) convert() error {
	if len(i.values) < 2 {
		return ErrNotEnoughData
	}
	if i.converted {
		return ErrAlreadyConverted
	}
	for j := range i.values {
		temp := i.values[j].y
		i.values[j].y = math.Log(temp)
	}
	i.converted = true
	return nil
}

func (o *Output) convert() error {
	if o.converted {
		return ErrAlreadyConverted
	}
	if o.err != nil {
		return ErrLinearRegression
	}
	o.converted = true
	o.a, o.b = math.Exp(o.a), math.Exp(o.b)
	return nil
}

func (r *Regression) Convert() error {
	return r.input.convert()
}

func (r *Regression) Run() error {
	if !r.input.converted {
		return ErrNotConverted
	}
	if r.ran {
		return ErrRanAlready
	}
	reg := regression.Regression{}
	for _, val := range r.input.values {
		reg.Train(regression.DataPoint(val.y,[]float64{val.x}))
	}
	err := reg.Run()
	if err != nil {
		r.output = Output{err: err}
		return ErrLinearRegression
	}
	r.output = Output{reg.Coeff(0),reg.Coeff(1), false, nil}
	r.ran = true
	return nil
}

func (i *Input) hasAlready(find float64) bool {
	for j := range i.covered {
		if i.covered[j] == find {
			return true
		}
	}
	return false
}

func (r *Regression) Result() (float64, float64, error) {
	if !r.ran {
		return 0,0,ErrNotRan
	}
	if err := r.output.convert(); err != nil {
		return 0,0,err
	}

	return r.output.a, r.output.b, nil
}