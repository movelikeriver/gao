// A simple implementation of Alternating Least Squares.
//
// Formula
//   R = P * Q
//
//  e_{ij}^2 = (r_{ij} - \sum_{k=1}^K{p_{ik}q_{kj}})^2 + \frac{\beta}{2} \sum_{k=1}^K{(||P||^2 + ||Q||^2)}
//  see https://www.latex4technics.com/creator.php?id=55a205aaede5f3.98624301&format=png&dpi=300&crop=1
//
//  p'_{ik} = p_{ik} + \alpha \frac{\partial}{\partial p_{ik}}e_{ij}^2 = p_{ik} + \alpha(2 e_{ij} q_{kj} - \beta p_{ik} )
//  see https://www.latex4technics.com/creator.php?id=55a1a0a73b8910.76933644&format=png&dpi=300&crop=1
//
//  q'_{kj} = q_{kj} + \alpha \frac{\partial}{\partial q_{kj}}e_{ij}^2 = q_{kj} + \alpha(2 e_{ij} p_{ik} - \beta q_{kj} )
//  see https://www.latex4technics.com/creator.php?id=55a20533a70f57.40146537&format=png&dpi=300&crop=1


package main

import (
	"fmt"
	"math/rand"
	"time"

	"github.com/gonum/matrix/mat64"
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

func main() {
	// 5 x 4 matrix
	arr := []float64{
		5, 3, 0, 1,
		4, 0, 0, 1,
		1, 1, 0, 5,
		1, 0, 0, 4,
		0, 1, 5, 4,
	}
	mr := mat64.NewDense(5, 4, arr)

	mm, nn := mr.Dims()
	const kn int = 2
	const alpha float64 = 0.0002
	const beta float64 = 0.02
	const iterNum int = 5000

	var mp *mat64.Dense
	var mq *mat64.Dense

	func() {
		r := rand.New(rand.NewSource(time.Now().UnixNano()))
		var arrp = make([]float64, mm*kn)
		for i := 0; i < len(arrp); i++ {
			arrp[i] = r.Float64()
		}
		var arrq = make([]float64, kn*nn)
		for i := 0; i < len(arrq); i++ {
			arrq[i] = r.Float64()
		}
		mp = mat64.NewDense(mm, kn, arrp)
		mq = mat64.NewDense(kn, nn, arrq)
	}()

	for iter := 0; iter < iterNum; iter++ {
		for i := 0; i < mm; i++ {
			for j := 0; j < nn; j++ {
			      	var tmp mat64.Dense
				tmp.Mul(mp, mq)
				var me mat64.Dense
				me.Sub(mr, &tmp)

				if mr.At(i, j) > 0 {
					ee := me.At(i, j)
					for k := 0; k < kn; k++ {
						mp.Set(i, k, mp.At(i, k)+alpha*(2*ee*mq.At(k, j)-beta*mp.At(i, k)))
						mq.Set(k, j, mq.At(k, j)+alpha*(2*ee*mp.At(i, k)-beta*mq.At(k, j)))
					}
				}
			}
		}

		fmt.Println("round:", iter)
		fmt.Println("mp:")
		printMatrix(mp)
		fmt.Println("mq:")
		printMatrix(mq)
	}

	fmt.Println("input: ")
	printMatrix(mr)
	fmt.Println("result: ")
	var tmp mat64.Dense
	tmp.Mul(mp, mq)
	printMatrix(&tmp)
}
