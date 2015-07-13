// Logistic regression naive grad descent implementation
//
// The LaTex format formula is:
//
// H_{\theta} ( \begin{bmatrix}X_{1},&...,&X_{N}\end{bmatrix})
// = 1 /
// (1 +
//  e^{ -(\begin{bmatrix}\theta_{1}\\...\\\theta_{N}\end{bmatrix} *
//        \begin{bmatrix}X_{1},&...,&X_{N}\end{bmatrix}) })
//
// https://www.latex4technics.com/creator.php?id=559e8f42f0d525.25273414&format=png&dpi=300&crop=1
//
// \begin{bmatrix}\theta_{1}\\...\\\theta_{N}\end{bmatrix}
// =
// \begin{bmatrix}\theta_{1}\\...\\\theta_{N}\end{bmatrix}
// -
// \alpha
// *
// \begin{bmatrix}X_{11},&...,&X_{M1}\\...\\X_{1N},&...,&X_{MN}\end{bmatrix}
// *
// \begin{bmatrix}
// H_{\theta} ( \begin{bmatrix}X_{11},&...,&X_{1N}\end{bmatrix}) - Y_{1}\\
// ...\\
// H_{\theta} ( \begin{bmatrix}X_{M1},&...,&X_{MN}\end{bmatrix}) - Y_{M}\\
// \end{bmatrix}
//
// https://www.latex4technics.com/creator.php?id=559e8c977df289.42728955&format=png&dpi=300&crop=1
//
// SGD (Stochastic Gradient Descent) formula is:
//
// \begin{bmatrix}\theta_{1}\\...\\\theta_{N}\end{bmatrix}
// = \begin{bmatrix}\theta_{1}\\...\\\theta_{N}\end{bmatrix}
// - \alpha
// * \begin{bmatrix}X_{k1}\\...\\X_{kN}\end{bmatrix}
// * ( H_{\theta} ( \begin{bmatrix}X_{k1},&...,&X_{kN}\end{bmatrix}) - Y_{k} )
//
// https://www.latex4technics.com/creator.php?id=55a3311268fea2.89861852&format=png&dpi=300&crop=1
//
// Usage:
//   go run logistic-reg.go --input_file=testdata1.txt --alpha=0.001 --iteration_num=2000

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/gonum/matrix/mat64"
)

var (
	inputFile = flag.String("input_file", "", "The input file.")
	alpha     = flag.Float64("alpha", 0.001, "The alpha step in each iteration.")
	iterNum   = flag.Int("iteration_num", 2000, "The max num of iteration.")
)

type fm struct {
	Matrix *mat64.Dense
	margin int
}

func (m fm) Format(fs fmt.State, c rune) {
	if c == 'v' && fs.Flag('#') {
		fmt.Fprintf(fs, "%#v", m.Matrix)
		return
	}
	mat64.Format(m.Matrix, m.margin, '.', fs, c)
}

func printMatrix(m *mat64.Dense) {
	mt := fm{Matrix: m, margin: 3}
	fmt.Printf("%v\n", mt)
}

func parseLine(line string) (arr []float64, y int) {
	segs := strings.Split(line, "\t")
	if len(segs) != 3 {
		fmt.Println(len(segs))
		return nil, 0
	}

	x1, err := strconv.ParseFloat(segs[0], 64)
	if err != nil {
		return nil, 0
	}
	x2, err := strconv.ParseFloat(segs[1], 64)
	if err != nil {
		return nil, 0
	}
	y, err = strconv.Atoi(segs[2])
	if err != nil {
		return nil, 0
	}

	// Prepend 1.0 into the training data.
	return []float64{1.0, x1, x2}, y
}

func sigmoid(theta *mat64.Dense, mx *mat64.Dense) float64 {
	var tmp mat64.Dense
	tmp.Mul(mx, theta)
	return 1.0 / (1.0 + math.Exp(-tmp.At(0, 0)))
}

func trainGradientDescent(alpha float64, iterNum int, mx *mat64.Dense, my *mat64.Dense, mtheta *mat64.Dense) {
	mm, nn := mx.Dims()
	// Sanity check.
	func() {
		yr, yc := my.Dims()
		tr, tc := mtheta.Dims()
		if yc != 1 || tc != 1 || yr != mm || tr != nn {
			log.Fatal("invalid dimensions of input matrixes.")
		}
	}()

	for it := 0; it < iterNum; it++ {
		tmp_arr := make([]float64, mm)
		for i := 0; i < mm; i++ {
			vx := mx.View(i, 0, 1, nn).(*mat64.Dense)
			sig := sigmoid(mtheta, vx)
			tmp_arr[i] = sig - my.At(i, 0)
		}
		tmp_matrix := mat64.NewDense(mm, 1, tmp_arr)

		var tmp1 mat64.Dense
		tmp1.MulTrans(mx, true, tmp_matrix, false)
		tmp1.Scale(alpha, &tmp1)
		mtheta.Sub(mtheta, &tmp1)

		fmt.Println("round: ", it)
		printMatrix(mtheta)
	}
}

func trainStochasticGradientDescent(alpha float64, iterNum int, mx *mat64.Dense, my *mat64.Dense, mtheta *mat64.Dense) {
	mm, nn := mx.Dims()
	// Sanity check.
	func() {
		yr, yc := my.Dims()
		tr, tc := mtheta.Dims()
		if yc != 1 || tc != 1 || yr != mm || tr != nn {
			log.Fatal("invalid dimensions of input matrixes.")
		}
	}()

	for it := 0; it < iterNum*mm; it++ {
		tmp_arr := make([]float64, 1)
		i := it % mm
		// SGD only pick up one sample data rather than the entire mm entries.
		vx := mx.View(i, 0, 1, nn).(*mat64.Dense)
		sig := sigmoid(mtheta, vx)
		tmp_arr[0] = sig - my.At(i, 0)
		tmp_matrix := mat64.NewDense(1, 1, tmp_arr)

		var tmp1 mat64.Dense
		tmp1.MulTrans(vx, true, tmp_matrix, false)

		tmp1.Scale(alpha, &tmp1)
		mtheta.Sub(mtheta, &tmp1)

		fmt.Println("round: ", it)
		printMatrix(mtheta)
	}
}

func statPrecision(mx *mat64.Dense, my *mat64.Dense, mtheta *mat64.Dense) float64 {
	rn, cn := mx.Dims()
	cnt := 0
	for i := 0; i < rn; i++ {
		vx := mx.View(i, 0, 1, cn).(*mat64.Dense)
		result := sigmoid(mtheta, vx)

		y := my.At(i, 0)
		fmt.Println(result, " vs ", y)
		if (y == 1 && result > 0.5) || (y == 0 && result < 0.5) {
			cnt++
		}
	}
	return float64(cnt) / float64(rn)
}

func main() {
	flag.Parse()

	f, err := os.Open(*inputFile)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	sarr := make([]float64, 0)
	sy := make([]float64, 0)
	r := bufio.NewReader(f)
	line, err := r.ReadString('\n')
	for err == nil {
		arr, y := parseLine(strings.TrimSpace(line))
		if arr != nil {
			sarr = append(sarr, arr...)
			sy = append(sy, float64(y))
		}
		line, err = r.ReadString('\n')
	}
	if err != io.EOF {
		log.Fatal(err)
	}

	nn := int(len(sarr) / len(sy)) // columns
	mm := len(sy)                  // rows
	mx := mat64.NewDense(mm, nn, sarr)
	my := mat64.NewDense(mm, 1, sy)
	ones := make([]float64, nn)
	for i := range ones {
		ones[i] = 1.0
	}
	mtheta := mat64.NewDense(nn, 1, ones)

	printMatrix(mx)
	printMatrix(my)
	printMatrix(mtheta)

	// trainGradientDescent(*alpha, *iterNum, mx, my, mtheta)
	trainStochasticGradientDescent(*alpha, *iterNum, mx, my, mtheta)

	fmt.Printf("precision: %.3f%%\n", statPrecision(mx, my, mtheta)*100)
}
