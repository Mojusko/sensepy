# Point process sensing library (sensepy)

This repository includes the code used in paper:

`Mojmir Mutny & Andreas Krause, "Sensing Cox Processes via Posterior Sampling and Positive Bases", AISTATS 2022`

For the paper see [here](https://mojmirmutny.github.io/#publications).

**Abstract**
 We study adaptive sensing of Cox point processes, a widely used model from spatial statistics. We introduce three tasks: maximization of captured events, search for the maximum of the intensity function and learning level sets of the intensity function. We model the intensity function as a sample from a truncated Gaussian process, represented in a specially constructed positive basis. In this basis, the positivity constraint on the intensity function has a simple form. We show how the \emph{minimal description positive basis} can be adapted to the covariance kernel, to non-stationarity and make connections to common positive bases from prior works. Our adaptive sensing algorithms use Langevin dynamics and are based on posterior sampling (\textsc{Cox-Thompson}) and top-two posterior sampling (\textsc{Top2}) principles. With latter, the difference between samples serves as a surrogate to the uncertainty. We demonstrate the approach using examples from environmental monitoring and crime rate modeling, and compare it to the classical Bayesian experimental design approach.


This **library in this repository** implements algorithms:

  - CaptureUCB [1]
  - Cox-Thompson [2]
  - V-optimal [2]
  - A modified version of Grant et. al. (2019) algorithm [3]
  - Top2 sampling [2]

**References**

1. Mutny M., Krause A., No-regret Algorithms for Capturing Events in Poisson Point Processes, ICML 2021
2. Mutny M., Krause A., Sensing Cox Processes via Posterior Sampling and Positive Bases, AISTATS 2022
3. Grant J. A. and Boukouvalas  A., and Griffiths R., Leslie D. Vakili S. and De Cote, Munoz, E. Adaptive sensor placement for continuous spaces. ICML 2019
## Installation
First clone the repository:

`git clone `

Inside the project directory, run

`pip install .`

The project requires Python 3.6+, and the dependencies should be installed with the package.

## Tutorials
We provide 2 tutorials:

1. Implementing sensing algorithms [Tutorial 1](https://github.com/Mojusko/sensepy/blob/master/sensepy/tutorial/implementing-sensing-algorithm.ipynb)
2. Fitting custom Cox process with geopandas maps [Tutorial 2](https://github.com/Mojusko/sensepy/blob/master/sensepy/tutorial/custom-kernel-cox-process.ipynb)

## Dependencies
  - Classical: torch, cvxpy, numpy, scipy, sklearn, pymanopt, mosek, pandas, geopandas

  - 1. stpy (see: <https://github.com/Mojusko/stpy>)
    2. pytorch-minimize <https://github.com/rfeinman/pytorch-minimize>

## Licence
MIT License

Copyright (c) 2022 Mojmir Mutny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Licence of datasets included

Datasets included in this repository stems from:

  1. **Taxi dataset** <https://www.kaggle.com/c/pkdd-15-predict-taxi-service-trajectory-i>
  2. **Gorillas and Beilschmiedia dataset** <https://spatstat.org/>

For licencing of the datasets, please refer to their original releases via the website above.

## Contributions
Mojmir Mutny, 2022
