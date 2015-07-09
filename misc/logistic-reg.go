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
// Usage:
//  go run logistic-reg.go --input_file=testSet.txt

package main

import (
	"bufio"
	"flag"
	"fmt"
	"io"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/gonum/matrix/mat64"
)

var inputFile = flag.String("input_file", "", "")

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

	return []float64{1.0, x1, x2}, y
}

func sigmoid(theta *mat64.Dense, mx *mat64.Dense) float64 {
	var tmp mat64.Dense
	tmp.Mul(mx, theta)
	return 1.0 / (1.0 + math.Exp(-tmp.At(0, 0)))
}

func main() {
	flag.Parse()

	f, err := os.Open(*inputFile)
	if err != nil {
		fmt.Println(err)
		return
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
		fmt.Println(err)
		return
	}

	nn := 3       // columns
	mm := len(sy) // rows
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

	alpha := 0.005
	for it := 0; it < 1000; it++ {
		func() {
			tmp_arr := make([]float64, mm)
			for i := 0; i < mm; i++ {
				vx := mx.View(i, 0, 1, nn).(*mat64.Dense)
				sig := sigmoid(mtheta, vx)
				tmp_arr[i] = (sig - my.At(i, 0)) * alpha
			}
			tmp_matrix := mat64.NewDense(mm, 1, tmp_arr)

			var tmp1 mat64.Dense
			tmp1.MulTrans(mx, true, tmp_matrix, false)
			mtheta.Sub(mtheta, &tmp1)

			fmt.Println("round: ", it)
			printMatrix(mtheta)
		}()
	}
}
