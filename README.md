TwoStageRidge
=============

A simple implementation of the two-stage ridge regression model described in
Hahn et. al (2018) with a scikit-learn compatible API.

> Hahn, P.R., Carvalho, C.M., Puelz, D., He, J., 2018. Regularization and
> Confounding in Linear Regression for Treatment Effect Estimation. Bayesian
> Anal. 13. https://doi.org/10.1214/16-BA1044

We have implemented maximum a-posteriori models rather than the fully Bayesian
treatment of the regression weights as described in (Hahn et. al, 2018). The
model implemented is;

1. Selection model: Z = **X**β<sub>c</sub> + ε,
2. Response model: Y = α(Z - **X**β<sub>c</sub>) + **X**β<sub>d</sub> + ν.

Here **X**, Y and Z are random variables. **X** are the controls, Z is the
treatment, and Y is the outcome. β<sub>c</sub> are first stage the linear
regression weights, β<sub>d</sub> are the second stage linear regression
weights on the control variables. α is the average treatment effect (ATE), and
ε ~ N(0, σ<sup>2</sup><sub>ε</sub>), ν ~ N(0, σ<sup>2</sup><sub>ν</sub>).

We place *l*<sub>2</sub> regularizers on the regression weights in the
regression objective functions,

1. Selection model: λ<sub>c</sub>·||β<sub>c</sub>||<sup>2</sup><sub>2</sub>,
2. Response model: λ<sub>d</sub>·||β<sub>d</sub>||<sup>2</sup><sub>2</sub>.

No regularisation is applied to α. This formulation leads to a less biased
estimation of α over alternate ridge regression models.

Installation
------------

This repository can be directly installed from GitHub, e.g.

    $ pip install git+git://github.com/gradientinstitute/twostageridge.git#egg=twostageridge

Quick start
-----------

`TwoStageRidge` uses a scikit learn interface. However, in order to retain
compatibility with all of the pipelines, model selection and other tool, we
have to treat the inputs to the model specially. That is, we have to
concatenate the control variables, `X` and the treatment variables `Z` into one
input array, e.g. `W = np.hstack((Z, X))`. For example,

```python
import numpy as np
from twostagerigde import TwoStageRidge

X, Y, Z = load_data()  # for some data function

# Where:
# - X.shape -> (N, D)
# - Y.shape -> (N,)
# - Z.shape -> (N,)

W = np.hstack((Z[:, np.newaxis], X))

ts = TwoStageRidge(treatment_index=0)  # Column index of the treatment variable
ts.fit(W, Y)  # estimate causal effect, alpha

print(f"α = {ts.alpha_}, s.e.(α) = {ts.se_alpha_}, p-value = {ts.p_}")
```

This will print out the estimated average treatment effect,standard error, and
p-value of a two-sided t-test against a null hypothesis of α = 0. For more
information on how to use this model, and how to perform model selection for
the model parameters, see the [notebooks](notebooks).


License
-------

Copyright 2021 Gradient Institute

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
