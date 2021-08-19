# Exponential Regression

## What's that?

Regression is finding an approximation to predict future and current data based on a set of given 
input data. Usually the approximation is put in a linear way, the so-called **Linear Regression**.

However, sometimes a simple line does not approximate the data's interaction and correlations properly.
In such cases one has to switch to a polynomial or exponential regression. The decision is based upon the
growth and growth expectance of the data. If this growth is in a similar extreme way as an exponential function 
(first a huge amount of almost no growth, then explosive growth) exponential regression should be the way.

## Example Usage

Caution: This example absolutely ignores error handling!

```go
package main
import (
    "fmt"

    "github.com/nillga/exponential-regression"
)

func main() {
    r := new(exponential-regression.Regression)
    r.Init([]struct{
        x float64
        y float64
    }{
        {1.0, 2.2},{1.1, 2.2},{1.2, 2.22},{1.3, 2.23},
        {1.4, 2.26},{1.5, 2.28},{1.6, 2.3},{1.7, 2.35},
        {1.8, 2.43},{1.9, 2.6},{2.0, 3.2},{2.1, 4.8},
    )
    r.Convert()
    r.Run()
    a,b,_ := r.Result()

    fmt.Printf("Regression formula:\ny=e^(%.4f+%.4f*x\n",a,b)
}
```

## Realisation

This package uses the base logic of this package: github.com/sajari/regression

It utilizes a linear regression only, but it is possible to calculate an exponential regression with the toolbox
of a linear regression. For doing this, a simple trick needs to be used.

### Linear Regression

Linear regression will output an equation of the scheme y = c + m * x, that is able to predict
values for x. As it is easily visible, this scheme is the scheme of a regular linear function.

### Exponential Regression

Therefore, for exponential regression a formula in an exponential scheme (y = a * b^x) is expected.
But how can we transform this into a linear equation? This is where the trick comes into play:

1.) We take the log of the equation: ln(y)=ln(a*b^x)

2.) By logarithm laws, we can drag this apart: ln(y)=ln(a)+ln(b^x)

3.) We can drag it further apart: ln(y)=ln(a)+x*ln(b)

ln(a) and ln(b) are constants, so we do end up with a linear equation.

4.) We perform a linear regression with the given x-values and the logarithm of our y-values

5.) LReg(x,ln(y)) ==> ln(y) = c + m*x

6.) Now we need to bring this back into shape: y = e^(c+m*x)

As easy as this, we have our exponential regression done.