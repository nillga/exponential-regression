package exponential_regression

import (
	"errors"
	"fmt"
	"math"
	"testing"
)

var (
	mockInput = []struct{
	x float64
	y float64
}{
	{1,2},{2,3},{3,4},
	}
	mockConv = []struct{
		x float64
		y float64
	}{
		{1, math.Log(2)},{2, math.Log(3)},{3, math.Log(4)},
	}
)

func TestRegression_Init (t *testing.T) {
	t.Run("Initialize empty", func(t *testing.T) {
		test := Regression{}
		err := test.Init(mockInput)
		if err != nil {
			t.Errorf("States initialization while there was none yet")
		}
		if test.input.converted {
			t.Errorf("Input field states conversion while there is none")
		}
		if index, neq := test.input.assert(mockInput); neq {
			t.Errorf("Error at index %d: Got {%.4f,%.4f} but wanted {%.4f,%.4f}", index,
				test.input.values[index].x,test.input.values[index].y, mockInput[index].x, mockInput[index].y)
		}
	})
	t.Run("Already initialized", func(t *testing.T) {
		test := Regression{initialized: true}
		err := test.Init(mockInput)
		if err == nil {
			t.Errorf("No error was thrown")
		}
		if err != ErrAlreadyInitialized {
			t.Errorf("Wrong error was thrown")
		}
	})
	t.Run("Negative Value", func(t *testing.T) {
		test := Regression{}
		err := test.Init([]struct {
			x float64
			y float64
		}{{1,-1}})

		assertError(err, ErrNegativeValue, t)
	})
}

func TestRegression_Append (t *testing.T) {
	t.Run("No Conversion yet", func(t *testing.T) {
		test := Regression{}
		err := test.Append(2.0,3.0)
		if err != nil {
			t.Errorf("Unexpected error was thrown: %v", err)
		}
		if test.input.values[0].x != 2.0 || test.input.values[0].y != 3.0 {
			t.Errorf("Appending wrong value: Got: {%.4f,%.4f} but expected: {2.0,3.0}", test.input.values[0].x,test.input.values[0].y)
		}
		if test.input.converted {
			t.Errorf("Input field states conversion while there is none")
		}
		if !test.initialized {
			t.Errorf("Initialization was not set true")
		}
	})
	t.Run("Converted already", func(t *testing.T) {
		test := Regression{}
		test.input.converted = true
		err := test.Append(2.0,3.0)
		if err == nil {
			t.Errorf("No error has been thrown")
		}
	})
	t.Run("Already initialized reg", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockInput)
		err := test.Append(4.0,5.0)

		if err != nil {
			t.Errorf("Wrongful error was thrown: %v", err)
		}
		i := len(test.input.values) - 1

		if test.input.values[i].x != 4.0 || test.input.values[i].y != 5.0 {
			t.Errorf("Appending wrong value: Got: {%.4f,%.4f} but expected: {4.0,5.0}", test.input.values[i].x,test.input.values[i].y)
		}
	})
	t.Run("Negative Value", func(t *testing.T) {
		test := Regression{}
		err := test.Append(5, -1)
		assertError(err, ErrNegativeValue, t)
	})
	t.Run("Multiple appearance of same x", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockInput)
		err := test.Append(3, 4)

		if err != nil {
			t.Errorf("Unexpected error occured: %v", err)
		}
		if len(test.input.values) != 3 {
			t.Errorf("Unexpected appending: Length increased to %d", len(test.input.values))
		}
	})
}

func TestInput_Convert (t *testing.T) {
	t.Run("optimal case", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockInput)
		input := test.input

		err := input.Convert()
		if err != nil {
			t.Errorf("Wrongful error was thrown")
		}
		if index, neq := input.assert(mockConv); neq {
			t.Errorf("Error at index %d: Got {%.4f,%.4f} but wanted {%.4f,%.4f}", index,
				test.input.values[index].x,test.input.values[index].y, mockConv[index].x, mockConv[index].y)
		}
		if !input.converted {
			t.Errorf("Conversion was not documented")
		}
	})
	t.Run("Little sample size", func(t *testing.T) {
		t.Run("No data", func(t *testing.T) {
			test := Input{}
			err := test.Convert()

			if err ==  nil {
				t.Errorf("No error was thrown")
			}
			if err != ErrNotEnoughData {
				t.Errorf("Wrong error was thrown: %v", err)
			}
		})
		t.Run("Not enough data", func(t *testing.T) {
			test := Input{values: []Value{{2,3}}}
			err := test.Convert()

			if err ==  nil {
				t.Errorf("No error was thrown")
			}
			if err != ErrNotEnoughData {
				t.Errorf("Wrong error was thrown: %v", err)
			}
		})
	})
	t.Run("Already converted data", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockInput)
		input := test.input
		input.converted = true

		err := input.Convert()
		assertError(err, ErrAlreadyConverted, t)
	})
}

func TestOutput_Convert (t *testing.T) {
	t.Run("optimal case", func(t *testing.T) {
		test := Output{}
		err := test.Convert()

		if err != nil {
			t.Errorf("Wrongful error appeared: %v", err)
		}
		if !test.converted {
			t.Errorf("Conversion is not marked")
		}
		if test.a != 1 || test.b != 1 {
			t.Errorf("Wrongful conversion: Expected a=1, b=1 but got a=%.2f, b=%.2f", test.a, test.b)
		}
	})
	t.Run("already converted", func(t *testing.T) {
		test := Output{converted: true}
		err := test.Convert()
		assertError(err, ErrAlreadyConverted, t)
	})
	t.Run("imported error", func(t *testing.T) {
		test := Output{err: errors.New("")}
		err := test.Convert()
		assertError(err, ErrLinearRegression, t)
	})
}

func TestFormulaToOutput (t *testing.T) {
	formulaWithEmptyVarName := fmt.Sprintf("Predicted = %.4f + %v*%.4f", 1.0,"X0",1.5)
	formulaWithBrokenC := fmt.Sprintf("Predicted = test + %v*%.4f", "X0",1.5)
	formulaWithBrokenM := fmt.Sprintf("Predicted = %.4f + %v*test", 1.0,"X0")
	t.Run("Optimal Case", func(t *testing.T) {
		got := FormulaToOutput(formulaWithEmptyVarName)
		want := Output{a:1,b:1.5}

		if *got != want {
			t.Errorf("Got: %v but expected: %v", got, want)
		}
	})
	t.Run("Error thrown", func(t *testing.T) {
		got := FormulaToOutput("")
		assertError(got.err, ErrInvalidFormula, t)
	})
	t.Run("Invalid Numbers", func(t *testing.T) {
		t.Run("a", func(t *testing.T) {
			got := FormulaToOutput(formulaWithBrokenC)
			assertError(got.err, ErrInvalidFormula, t)
		})
		t.Run("b", func(t *testing.T) {
			got := FormulaToOutput(formulaWithBrokenM)
			assertError(got.err, ErrInvalidFormula, t)
		})
	})
}

func TestRegression_Run (t *testing.T) {
	t.Run("Optimal Case", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockInput)
		_ = test.Convert()

		err := test.Run()

		if err != nil {
			t.Errorf("Unexpected error was thrown: %v", err)
		}
		if test.output.err != nil {
			t.Errorf("Unexpected error was thrown: %v", test.output.err)
		}
	})
	t.Run("Non converted Values", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockInput)

		err := test.Run()

		assertError(err, ErrNotConverted, t)
	})
	t.Run("RanAlready", func(t *testing.T) {
		test := Regression{ran: true, input: Input{converted: true}}
		err := test.Run()

		assertError(err,ErrRanAlready,t)
	})
	t.Run("Fail Linear", func(t *testing.T) {
		test := Regression{input: Input{converted: true}}
		err := test.Run()
		assertError(err, ErrLinearRegression, t)
	})

	t.Run("Example", func(t *testing.T){
		r := Regression{}
		r.Init([]struct{
			x float64
			y float64
		}{
			{1.0, 2.2},{1.1, 2.2},{1.2, 2.22},{1.3, 2.23},
			{1.4, 2.26},{1.5, 2.28},{1.6, 2.3},{1.7, 2.35},
			{1.8, 2.43},{1.9, 2.6},{2.0, 3.2},{2.1, 4.8},
		})
		r.Convert()
		r.Run()
		a,b,_ := r.Result()

		t.Logf("Regression formula:\ny=%.4f*%.4f^x\n",a,b)
	})
}

func TestInput_hasAlready (t *testing.T) {
	temp := Regression{}
	_ = temp.Init(mockInput)

	test := temp.input
	t.Run("find", func(t *testing.T) {
		got := test.hasAlready(2.0)

		if !got {
			t.Errorf("Did not find existing value 2.0")
		}
	})
	t.Run("not find", func(t *testing.T) {
		got := test.hasAlready(20.0)

		if got {
			t.Errorf("Found what was not there")
		}
	})
}

func TestRegression_Result(t *testing.T) {
	t.Run("Optimal Case", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockInput)
		_ = test.Convert()
		_ = test.Run()
		_,_,err := test.Result()
		if err != nil {
			t.Errorf("Unexpected Error occured: %v", err)
		}
	})
	t.Run("Not Run yet", func(t *testing.T) {
		test := Regression{}
		_,_,err := test.Result()

		assertError(err, ErrNotRan, t)
	})
	t.Run("Converting Issues", func(t *testing.T) {
		test := Regression{ran: true}
		t.Run("already converted", func(t *testing.T) {
			test.output = Output{converted: true}
			_,_,err := test.Result()
			assertError(err, ErrAlreadyConverted, t)
		})
		t.Run("imported error", func(t *testing.T) {
			test.output = Output{err: errors.New("")}
			_,_,err := test.Result()
			assertError(err, ErrLinearRegression, t)
		})
	})
}

func (i *Input) assert (s []struct{
	x float64
	y float64
}) (int, bool) {
	for j := range i.values {
		if i.values[j].x != s[j].x {
			return j, true
		}
		if i.values[j].y != s[j].y {
			return j, true
		}
	}
	return 0, false
}

func assertError(targ, ideal error, t *testing.T) {
	if targ == nil {
		t.Errorf("No error was thrown")
	}
	if targ != ideal {
		t.Errorf("Wrongful error was thrown: %v", targ)
	}
}