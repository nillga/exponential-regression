package exponential_regression

import (
	"errors"
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

	mockMap = map[float64]float64{
		1.0:2.0,
		2.0:3.0,
		3.0:4.0,
	}
)

func TestRegression_Init (t *testing.T) {
	t.Run("Initialize empty", func(t *testing.T) {
		test := Regression{}
		err := test.Init(mockMap)
		if err != nil {
			t.Errorf("States initialization while there was none yet")
		}
		if test.input.converted {
			t.Errorf("Input field states conversion while there is none")
		}
		test.input.assert(mockInput, t)
	})
	t.Run("Already initialized", func(t *testing.T) {
		test := Regression{initialized: true}
		err := test.Init(mockMap)
		if err == nil {
			t.Errorf("No error was thrown")
		}
		if err != ErrAlreadyInitialized {
			t.Errorf("Wrong error was thrown")
		}
	})
	t.Run("Negative Value", func(t *testing.T) {
		test := Regression{}
		err := test.Init(map[float64]float64{0.0:-1.0})

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
		_ = test.Init(mockMap)
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
		_ = test.Init(mockMap)
		err := test.Append(3, 4)

		if err != nil {
			t.Errorf("Unexpected error occured: %v", err)
		}
		if len(test.input.values) != 3 {
			t.Errorf("Unexpected appending: Length increased to %d", len(test.input.values))
		}
	})
}

func TestInput_convert (t *testing.T) {
	t.Run("optimal case", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockMap)
		input := test.input

		err := input.convert()
		if err != nil {
			t.Errorf("Wrongful error was thrown")
		}
		if !input.converted {
			t.Errorf("Conversion was not documented")
		}
		input.assert(mockConv, t)
	})
	t.Run("Little sample size", func(t *testing.T) {
		t.Run("No data", func(t *testing.T) {
			test := Input{}
			err := test.convert()

			if err ==  nil {
				t.Errorf("No error was thrown")
			}
			if err != ErrNotEnoughData {
				t.Errorf("Wrong error was thrown: %v", err)
			}
		})
		t.Run("Not enough data", func(t *testing.T) {
			test := Input{values: []Value{{2,3}}}
			err := test.convert()

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
		_ = test.Init(mockMap)
		input := test.input
		input.converted = true

		err := input.convert()
		assertError(err, ErrAlreadyConverted, t)
	})
}

func TestOutput_convert (t *testing.T) {
	t.Run("optimal case", func(t *testing.T) {
		test := Output{}
		err := test.convert()

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
		err := test.convert()
		assertError(err, ErrAlreadyConverted, t)
	})
	t.Run("imported error", func(t *testing.T) {
		test := Output{err: errors.New("")}
		err := test.convert()
		assertError(err, ErrLinearRegression, t)
	})
}

func TestRegression_Run (t *testing.T) {
	t.Run("Optimal Case", func(t *testing.T) {
		test := Regression{}
		_ = test.Init(mockMap)
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
		_ = test.Init(mockMap)

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
		r.Init(map[float64]float64{
			1.0: 2.2,
			1.1: 2.2,
			1.2: 2.22,
			1.3: 2.23,
			1.4: 2.26,
			1.5: 2.28,
			1.6: 2.3,
			1.7: 2.35,
			1.8: 2.43,
			1.9: 2.6,
			2.0: 3.2,
			2.1: 4.8,
		})
		r.Convert()
		r.Run()
		a,b,_ := r.Result()

		t.Logf("Regression formula:\ny=%.4f*%.4f^x\n",a,b)
	})
}

func TestInput_hasAlready (t *testing.T) {
	temp := Regression{}
	_ = temp.Init(mockMap)

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
		_ = test.Init(mockMap)
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
}, t *testing.T) {
	for j := range i.values {
		if index, problem := i.find(s[j].x,s[j].y); problem {
			if index < 0 {
				t.Errorf("Error: Missing Fields Got %v but wanted %v",i.values,s)
				return
			}
			t.Errorf("Error at index %d: Got {%.4f,%.4f} but wanted {%.4f,%.4f}", index,
				i.values[index].x,i.values[index].y, s[index].x, s[index].y)
		}
	}
}

func (i *Input) find (x,y float64) (int,bool) {
	for j := range i.values {
		if i.values[j].x == x {
			if i.values[j].y != y {
				return j, true
			}
			return 0,false
		}
	}
	return 0, true
}

func assertError(targ, ideal error, t *testing.T) {
	if targ == nil {
		t.Errorf("No error was thrown")
	}
	if targ != ideal {
		t.Errorf("Wrongful error was thrown: %v", targ)
	}
}