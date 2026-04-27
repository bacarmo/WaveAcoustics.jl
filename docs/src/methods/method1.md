# Scheme 1
We first present a numerical scheme based on the Crank-Nicolson Galerkin method, which consists of finding ``U^n, V^n \in \mathcal{V}_{m_1}`` and ``Z^n, R^n \in \mathcal{V}_{m_2}`` such that
```math
\begin{align}
\label{def:approx1}
\begin{aligned}
& \big(\varphi,\bar{\partial}V^n\big)
+ \alpha^{n-\frac{1}{2}}\Big[
    \big(\nabla\varphi,\nabla\widehat{U}^n\big)
  - \big(\varphi,\widehat{R}^n)_{\Gamma_1}
  + \big(\varphi,g(\widehat{V}^n)\big)_{\Gamma_1}\Big]
+ \big(\varphi,f(\widehat{U}^n)\big) 
= \big(\varphi,f_1^{n-\frac{1}{2}}\big),
\quad\forall\varphi\in \mathcal{V}_{m_1},
\\
& \big(\phi,q_1\bar{\partial}R^n
+ q_2\widehat{R}^n
+ q_3\widehat{Z}^n
+ q_4\widehat{V}^n\big)_{\Gamma_1}
= \big(\phi,f_2^{n-\frac{1}{2}}\big)_{\Gamma_1},
\quad\forall\phi\in \mathcal{V}_{m_2},
\\
& 
\bar{\partial}U^n = \widehat{V}^n,\quad 
\bar{\partial}Z^n = \widehat{R}^n,
\end{aligned}
\end{align}
```
with ``U^0, V^0 \in \mathcal{V}_{m_1}`` and ``Z^0, R^0 \in \mathcal{V}_{m_2}`` given as approximations of the initial solutions ``u_0, v_0, z_0``, and ``r_0``.
We define these approximations by solving the following projections problems:
find  ``U^0,\, V^0 \in \mathcal{V}_{m_1}``, and ``Z^0,\,R^0 \in \mathcal{V}_{m_2}`` such that
```math
\begin{align*}
& \big(\nabla\varphi,\nabla U^0\big) 
= \big(\nabla\varphi,\nabla u_0\big), \;\;
\forall\varphi\in \mathcal{V}_{m_1};
\\
& \big(\nabla\varphi,\nabla V^0\big) 
= \big(\nabla\varphi,\nabla v_0\big),\;\;
\forall\varphi\in \mathcal{V}_{m_1};
\\
& \big(\phi,Z^0\big)_{\Gamma_1} 
= \big(\phi,z_0\big)_{\Gamma_1},\;\;
\forall\phi\in \mathcal{V}_{m_2};
\\
& \big(\phi,R^0\big)_{\Gamma_1} 
= \big(\phi,r_0\big)_{\Gamma_1},\;\;
\forall\phi\in \mathcal{V}_{m_2}.
\end{align*}
```

!!! details "Notation"
    - ``\mathcal{V}_{m_1} \subset H_{\Gamma_0}^1(\Omega)``: Subspace of dimension ``m_1`` with basis ``\{\varphi_j\}_{j=1}^{m_1}``.
    - ``\mathcal{V}_{m_2} = \mathcal{V}_{m_1}|_{\Gamma_1}``: Subspace of dimension ``m_2`` with basis ``\{\phi_j\}_{j=1}^{m_2}`` satisfying
    ```math
        \phi_j = \varphi_j|_{\Gamma_1}, \quad j = 1, \dots, m_2.
    ```
    - ``\displaystyle w^n:=w(t_n),\quad \bar{\partial}w^n := \frac{w^n - w^{n-1}}{\tau}\approx w^\prime(t_{n-\frac{1}{2}}),\quad \widehat{w}^n := \frac{w^n + w^{n-1}}{2}\approx w(t_{n-\frac{1}{2}}),``
    where ``\tau`` denotes the time step, ``t_n = n\tau`` the discrete times, ``t_{n-\frac{1}{2}}`` the midpoint of  the interval ``[t_{n-1},t_{n}]``, and ``w`` an arbitrary time-dependent function.
    
## Matrix Formulation
Representing the approximate solutions in terms of the basis functions,
```math
U^n = \sum_{j=1}^{m_1} d_j^n\varphi_j,\;
V^n = \sum_{j=1}^{m_1} v_j^n\varphi_j,\;
Z^n = \sum_{j=1}^{m_2} z_j^n\phi_j,\;
R^n = \sum_{j=1}^{m_2} r_j^n\phi_j,
```
and choosing test functions ``\varphi=\varphi_i`` for ``i = 1, \ldots, m_1`` and ``\phi=\phi_i`` for ``i = 1, \ldots, m_2``, we obtain the system
```math
\begin{align}
\label{def:approx1:mat_form}
\begin{aligned}
& M^{m_1\times m_1}\bar{\partial}v^n
+ \alpha^{n-\frac{1}{2}}\Big[  
    K^{m_1\times m_1}\widehat{d}^n
  - M^{m_1\times m_2}\widehat{r}^n
  + G^{m_1}(\widehat{v}^n)\Big]
+ F^{m_1}(\widehat{d}^n)
= \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}),
\\
& M^{m_2\times m_2}\big[
  q_1\bar{\partial}r^n
+ q_2\widehat{r}^n
+ q_3\widehat{z}^n]
+ q_4M^{m_2\times m_1}\widehat{v}^n
= \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}),
\\
& \bar{\partial}d^n = \widehat{v}^n,\quad
\bar{\partial}z^n = \widehat{r}^n,
\end{aligned}
\end{align}
```
with initial solution given by
```math
K^{m_1\times m_1} d^0 = \kappa^{m_1}(u_0),\quad
K^{m_1\times m_1} v^0 = \kappa^{m_1}(v_0),\quad
M^{m_2\times m_2} z^0 = \mathcal{F}^{m_2}(z_0),\quad
M^{m_2\times m_2} r^0 = \mathcal{F}^{m_2}(r_0).
```

!!! details "Matrix and vector definitions"
    ```math
    \begin{aligned}
    &
    M^{m_1\times m_1}_{i,j} = (\varphi_i,\varphi_j),\quad 
    M^{m_1\times m_2}_{i,j} = (\varphi_i,\phi_j)_{\Gamma_1},\quad 
    M^{m_2\times m_1}_{i,j} = (\phi_i,\varphi_j)_{\Gamma_1},\quad 
    M^{m_2\times m_2}_{i,j} = (\phi_i,\phi_j)_{\Gamma_1},
    \\[5pt]
    &
    K^{m_1\times m_1}_{i,j} = (\nabla\varphi_i,\nabla\varphi_j),\quad
    \mathcal{F}_i^{m_1}(\omega) = \big(\varphi_i,\omega\big),\quad
    \mathcal{F}_i^{m_2}(\omega) = \big(\phi_i,\omega\big)_{\Gamma_1},\quad
    \kappa_i^{m_1}(\omega) =  \big(\nabla\varphi_i,\nabla\omega\big),
    \\[5pt]
    &
    G^{m_1}_i(\widehat{v}^n) 
    = \big(\varphi_i,g(\widehat{V}^n)\big)_{\Gamma_1}
    \equiv\int_{\Gamma_1}\varphi_i(x)g\Big(x,\sum_{\ell=1}^{m_1}\widehat{v}_\ell^n\varphi_\ell(x)\Big)d\Gamma,
    \\[5pt]
    &
    F_i^{m_1}(\widehat{d}^n) 
    = \big(\varphi_i,f(\widehat{U}^n)\big)
    \equiv\int_{\Omega}\varphi_i(x)f\Big(\sum_{\ell=1}^{m_1}\widehat{d}_\ell^n\varphi_\ell(x)\Big)dx.
    \end{aligned}
    ```

## Solving the Nonlinear Algebraic System
The matrix formulation is equivalent to finding
``X`` such that ``H(X) = 0``, 
which we solve via Newton's method.
Starting from the initial guess 
``X_0 = v^{n-1}``, 
the method generates the sequence 
``X_{\kappa+1} = X_\kappa + S_\kappa``, 
where ``S_\kappa`` is obtained by solving the linear system
```math
JH(X_\kappa)\, S_\kappa = -H(X_\kappa),
```
with ``JH(X_\kappa)`` denoting the Jacobian of ``H`` evaluated at ``X_\kappa``.

We consider three strategies to formulate ``H(X) = 0``, differing in the choice of primary unknowns.

### Strategy 1: Formulation in Terms of ``X=v^n``
```math
Q(n)v^n
+ \tau\alpha^{n-\frac{1}{2}} G^{m_1}\big(\frac{v^n+v^{n-1}}{2}\big) 
 + \tau F^{m_1}\big(\frac{\tau}{4}v^{n}+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
 - L(n)
= 0.
```
Once ``v^n`` has been determined, we compute ``\hat{r}^n`` via
```math
\hat{r}^n
=
- \frac{\tau q_4}{q_5}\hat{v}_{1:m_2}^n
+ \frac{2q_1}{q_5} r^{n-1}
- \frac{\tau q_3}{q_5} z^{n-1}
+ \frac{\tau}{q_5} \Big(M^{m_2\times m_2}\Big)^{-1}
  \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}),
```
and the remaining quantities ``d^n``, ``z^n``, and ``r^n`` by
```math
\begin{align*}
&     r^n = 2\hat{r}^n - r^{n-1},
\quad d^n = \tau\hat{v}^n + d^{n-1},
\quad z^n = \tau\hat{r}^n + z^{n-1}.
\end{align*}
```

!!! details "Details"
    From the matrix formulation, we rewrite the second equation in terms of ``\hat{r}^n`` and substitute it into the first:
    ```math
    \begin{align*}
    & M^{m_1\times m_1}\bar{\partial}v^n
    + \alpha^{n-\frac{1}{2}}\Big[  
        K^{m_1\times m_1}\widehat{d}^n
      + G^{m_1}(\widehat{v}^n)\Big]
    + F^{m_1}(\widehat{d}^n)
    \\[5pt]
    &\qquad
    + \frac{\alpha^{n-\frac{1}{2}}}{q_5}
    \begin{bmatrix}
      \tau q_4M^{m_2\times m_2} \widehat{v}_{1:m_2}^n
      - M^{m_2\times m_2}\big( 2q_1r^{n-1} - \tau q_3z^{n-1} \big)
      - \tau \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}})
      \\[5pt]
      0^{(m_1-m_2)}
    \end{bmatrix}
    = \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}).
    \end{align*}
    ```
    Then, isolating ``v^n``: 
    ```math
    \begin{align*}
    &
    \Big(
        M^{m_1\times m_1}
        + \frac{\tau^2}{4}\alpha^{n-\frac{1}{2}} K^{m_1\times m_1}
        + \frac{\tau^2q_4}{2q_5}\alpha^{n-\frac{1}{2}}
          \begin{bmatrix}
          M^{m_2\times m_2}       & 0^{m_2\times(m_1-m_2)}\\[5pt]
          0^{(m_1-m_2)\times m_2} & 0^{(m_1-m_2)\times(m_1-m_2)}
          \end{bmatrix}
    \Big) v^n
    \\[10pt]
    &\qquad
    + \tau\alpha^{n-\frac{1}{2}} G^{m_1}(\frac{v^n+v^{n-1}}{2}) 
    + \tau F^{m_1}(\frac{\tau}{4}v^n+\frac{\tau}{4}v^{n-1} + d^{n-1})
    - L(n) = 0.
    \end{align*}
    ```
    !!! note "Beware!"
        To rewrite the second equation in terms of ``\hat{r}^n``, we use the identities:
        ```math
        \bar\partial r^n = \frac{2}{\tau}\widehat{r}^n - \frac{2}{\tau}z^{n-1},
        \quad
        \widehat{z}^n = \frac{\tau}{2} \widehat{r}^n + z^{n-1}.
        ```

!!! details "Matrix and vector definitions"
    ```math
    \begin{align*}
    Q(n) =& 
    M^{m_1\times m_1} 
    + \frac{\tau^2}{4}\alpha^{n-\frac{1}{2}}K^{m_1\times m_1}
    + \frac{\tau^2q_4}{2q_5}\alpha^{n-\frac{1}{2}}
    \begin{bmatrix}
    M^{m_2\times m_2}       & 0^{m_2\times(m_1-m_2)}\\[5pt]
    0^{(m_1-m_2)\times m_2} & 0^{(m_1-m_2)\times(m_1-m_2)}
    \end{bmatrix}
    \\[30pt]
    L(n) =&
    M^{m_1\times m_1}v^{n-1}
    - \tau \alpha^{n-\frac{1}{2}}K^{m_1\times m_1} (\frac{\tau}{4}v^{n-1}+d^{n-1})
    + \tau \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}})
    \\[10pt]
    &
    + \frac{\tau}{q_5}\alpha^{n-\frac{1}{2}}
    \begin{bmatrix}
      M^{m_2\times m_2}\big(-\frac{\tau}{2}q_4v_{1:m_2}^{n-1} + 2q_1r^{n-1} - \tau q_3z^{n-1} \big)
      + \tau \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}})
      \\[5pt]
      0^{(m_1-m_2)}
    \end{bmatrix}
    \end{align*}
    ```
    where ``q_5 = 2q_1+\tau q_2+\frac{\tau^2}{2}q_3``.


!!! details " Jacobian matrix calculation"
    Initially, note that:
    ```math
    H_i(X) = \sum_{\ell=1}^{m_1}Q_{i,\ell}X_\ell
    + 
    \tau\alpha^{n-\frac{1}{2}}
    G_i^{m_1}\big(\frac{X+v^{n-1}}{2}\big) 
    + \tau F_i^{m_1}\big(\frac{\tau}{4}X+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
    - L_i.
    ```
    In this way,
    ```math
    JH_{i,j}(X)
    = \frac{\partial H_i}{\partial X_j}(X) 
    = Q_{i,j} 
    + \frac{\tau\alpha^{n-\frac{1}{2}}}{2}JG_{i,j}\big(\frac{X+v^{n-1}}{2}\big) 
    + \frac{\tau^2}{4} JF_{i,j}\big(\frac{\tau}{4}X+\frac{\tau}{4}v^{n-1}+d^{n-1}\big),
    ```
    where
    ```math
    G_i^{m_1}(v) 
    = \int_{\Gamma_1}\varphi_i(x)g\big(x,\sum_{\ell=1}^{m_1}v_\ell\varphi_\ell(x)\big)d\Gamma
    \;
    \Rightarrow
    \;
    \frac{\partial}{\partial X_j} G_i^{m_1}\big(\frac{X+v^{n-1}}{2}\big)
    = \frac{1}{2}\int_{\Gamma_1}\varphi_i(x)\varphi_j(x)\frac{\partial g}{\partial s}\big(x,\sum_{\ell=1}^{m_1}\frac{X_\ell+v_{\ell}^{n-1}}{2}\varphi_\ell(x)\big)d\Gamma
    = \frac{1}{2} JG_{i,j}\big(\frac{X+v^{n-1}}{2}\big)
    ```
    and
    ```math
    F_i^{m_1}(d) 
    = 
    \int_{\Omega}\varphi_i(x)f\Big(\sum_{\ell=1}^{m_1}d_\ell\varphi_\ell(x)\Big)dx
    \;
    \Rightarrow
    \;
    \frac{\partial}{\partial X_j}F_i^{m_1}\big(\frac{\tau}{4}X+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
    = 
    \frac{\tau}{4}\int_{\Omega}\varphi_i(x)\varphi_j(x)f^\prime\Big(\sum_{\ell=1}^{m_1}
    \big[\frac{\tau}{4}X_\ell+\frac{\tau}{4}v_\ell^{n-1}+d_\ell^{n-1}\big]
    \varphi_\ell(x)\Big)dx
    = \frac{\tau}{4} JF_{i,j}\big(\frac{\tau}{4}X+\frac{\tau}{4}v^{n-1}+d^{n-1}\big).
    ```
    !!! note "Beware!"
        ```math
        JG^{m_1\times m_1}(v) =
        \begin{bmatrix}\displaystyle
        JG^{m_2\times m_2}(v_{1:m_2}) & 0^{m_2\times(m_1-m_2)}
        \\[10pt]
        0^{(m_1-m_2)\times m_2}       & 0^{(m_1-m_2)\times(m_1-m_2)}
        \end{bmatrix}
        ```
### Strategy 2: Formulation in Terms of ``X=\hat{v}^n``
```math
Q(n) \hat{v}^n
+ \tau\alpha^{n-\frac{1}{2}} G^{m_1}(\widehat{v}^n) 
+ \tau F^{m_1}(\frac{\tau}{2}\widehat{v}^n + d^{n-1})
- L(n)
= 0.
```
Once ``\hat{v}^n`` has been determined, we compute ``\hat{r}^n`` via
```math
\hat{r}^n
=
- \frac{\tau q_4}{q_5}\hat{v}_{1:m_2}^n
+ \frac{2q_1}{q_5} r^{n-1}
- \frac{\tau q_3}{q_5} z^{n-1}
+ \frac{\tau}{q_5} \Big(M^{m_2\times m_2}\Big)^{-1}
  \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}),
```
and the remaining quantities ``d^n``, ``v^n``, ``z^n``, and ``r^n`` are then recovered by
```math
\begin{align*}
& v^n = 2\hat{v}^n - v^{n-1},
\quad r^n = 2\hat{r}^n - r^{n-1},
\quad d^n = \tau\hat{v}^n + d^{n-1},
\quad z^n = \tau\hat{r}^n + z^{n-1}.
\end{align*}
```

!!! details "Details"
    Rewriting the second equation in terms of ``\hat{r}^n`` and substituting it into the first yields:
    ```math
    \begin{align*}
    & M^{m_1\times m_1}\bar{\partial}v^n
    + \alpha^{n-\frac{1}{2}}\Big[  
        K^{m_1\times m_1}\widehat{d}^n
      + G^{m_1}(\widehat{v}^n)\Big]
    + F^{m_1}(\widehat{d}^n)
    \\[5pt]
    &\qquad
    + \frac{\alpha^{n-\frac{1}{2}}}{q_5}
    \begin{bmatrix}
      \tau q_4M^{m_2\times m_2} \widehat{v}_{1:m_2}^n
      - M^{m_2\times m_2}\big( 2q_1r^{n-1} - \tau q_3z^{n-1} \big)
      - \tau \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}})
      \\[5pt]
      0^{(m_1-m_2)}
    \end{bmatrix}
    = \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}).
    \end{align*}
    ```
    Applying the identities
    ```math
    \bar\partial v^n = \frac{2}{\tau}\widehat{v}^n - \frac{2}{\tau}v^{n-1},
    \quad
    \widehat{d}^n = \frac{\tau}{2} \widehat{v}^n + d^{n-1},
    ```
    we obtain
    ```math
    \begin{align*}
    & M^{m_1\times m_1} \big(\frac{2}{\tau}\widehat{v}^n-\frac{2}{\tau}v^{n-1}\big)
    + \alpha^{n-\frac{1}{2}}\Big[  
        K^{m_1\times m_1} \big(\frac{\tau}{2}\widehat{v}^n+d^{n-1}\big)
      + G^{m_1}(\widehat{v}^n)\Big]
    + F^{m_1}(\frac{\tau}{2}\widehat{v}^n + d^{n-1})
    \\[10pt]
    &\qquad
    + \frac{\alpha^{n-\frac{1}{2}}}{q_5}
    \begin{bmatrix}
      \tau q_4M^{m_2\times m_2} \widehat{v}_{1:m_2}^n
      - M^{m_2\times m_2}\big( 2q_1r^{n-1} - \tau q_3z^{n-1} \big)
      - \tau \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}})
      \\[5pt]
      0^{(m_1-m_2)}
    \end{bmatrix}
    = \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}).
    \end{align*}
    ```
    Isolating ``\widehat{v}^n``:
    ```math
    \begin{align*}
    &
    \Big(
        2M^{m_1\times m_1}
        + \frac{\tau^2}{2}\alpha^{n-\frac{1}{2}} K^{m_1\times m_1}
        + \frac{\tau^2q_4}{q_5}\alpha^{n-\frac{1}{2}}
          \begin{bmatrix}
          M^{m_2\times m_2}       & 0^{m_2\times(m_1-m_2)}\\[5pt]
          0^{(m_1-m_2)\times m_2} & 0^{(m_1-m_2)\times(m_1-m_2)}
          \end{bmatrix}
    \Big) \widehat{v}^n
    \\[10pt]
    &\qquad
    + \tau\alpha^{n-\frac{1}{2}} G^{m_1}(\widehat{v}^n) 
    + \tau F^{m_1}(\frac{\tau}{2}\widehat{v}^n + d^{n-1})
    - L(n) = 0.
    \end{align*}
    ```

!!! details "Matrix and vector definitions"
    ```math
    \begin{align*}
    Q(n) = & 
    2M^{m_1\times m_1} 
    + \frac{\tau^2}{2}\alpha^{n-\frac{1}{2}}K^{m_1\times m_1}
    + \frac{\tau^2q_4}{q_5}\alpha^{n-\frac{1}{2}}
    \begin{bmatrix}
    M^{m_2\times m_2}       & 0^{m_2\times(m_1-m_2)}\\[5pt]
    0^{(m_1-m_2)\times m_2} & 0^{(m_1-m_2)\times(m_1-m_2)}
    \end{bmatrix}
    \\[30pt]
    L(n) =&
    2M^{m_1\times m_1}v^{n-1}
    - \tau \alpha^{n-\frac{1}{2}}K^{m_1\times m_1} d^{n-1}
    + \tau \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}})
    + \frac{\tau}{q_5}\alpha^{n-\frac{1}{2}}
    \begin{bmatrix}
      M^{m_2\times m_2}\big( 2q_1r^{n-1} - \tau q_3z^{n-1} \big)
      + \tau \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}})
      \\[5pt]
      0^{(m_1-m_2)}
    \end{bmatrix}
    \end{align*}
    ```
!!! details " Jacobian matrix calculation"
    ```math
    JH_{i,j}(X)
    = \frac{\partial H_i}{\partial X_j}(X) 
    = Q_{i,j} 
    + \tau\alpha^{n-\frac{1}{2}}JG_{i,j}\big(X\big) 
    + \frac{\tau^2}{2} JF_{i,j}\big(\frac{\tau}{2}X+d^{n-1}\big).
    ```

### Strategy 3: Formulation in Terms of ``X=\begin{bmatrix}v^n\\r^n\end{bmatrix}``
```math
\label{prob1:nonlinear_system}
\begin{aligned}
Q(n)
\begin{bmatrix}
 v^n \\[5pt]
 r^n
\end{bmatrix}
+ 
\begin{bmatrix}
 \tau\alpha^{n-\frac{1}{2}}
 G^{m_1}\big(\frac{v^n+v^{n-1}}{2}\big) 
 + \tau F^{m_1}\big(\frac{\tau}{4}v^{n}+\frac{\tau}{4}v^{n-1}+d^{n-1}\big) \\[5pt]
 0^{m_2}
\end{bmatrix}
-
\begin{bmatrix}
L_1(n)\\
L_2(n)
\end{bmatrix}
= 0.
\end{aligned}
```
Once ``v^n`` and ``c^n`` have been determined, we compute ``d^n`` and ``z^n`` by
```math
d^n = d^{n-1} + \frac{\tau}{2}(v^n+v^{n-1}), \quad
z^n = z^{n-1} + \frac{\tau}{2}(r^n+r^{n-1}).
```

!!! details "Matrix and vector definitions"
    ```math
    \begin{aligned}
    &Q =
    \begin{bmatrix}
      M^{m_1\times m_1} + \frac{\tau^2\alpha^{n-\frac{1}{2}}}{4}K^{m_1\times m_1} 
    &-\frac{\tau\alpha^{n-\frac{1}{2}}}{2}M^{m_1\times m_2}  
    \\[5pt]
      \frac{\tau}{2}q_4M^{m_2\times m_1}
    & (q_1 + \frac{\tau}{2}q_2 + \frac{\tau^2}{4}q_3)M^{m_2\times m_2}
    \end{bmatrix},
    \\[10pt]
    &L_1 = 
    M^{m_1\times m_1}v^{n-1}
    - \tau\alpha^{n-\frac{1}{2}} K^{m_1\times m_1}\Big(\frac{\tau}{4}v^{n-1}+d^{n-1}\Big)
    + \frac{\tau}{2}\alpha^{n-\frac{1}{2}}M^{m_1\times m_2}r^{n-1}
    + \tau\mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}),
    \\[10pt]
    &L_2 =
    M^{m_2\times m_2}\Big[
      (q_1 - \frac{\tau}{2}q_2 - \frac{\tau^2}{4}q_3)r^{n-1}
      - \tau q_3 z^{n-1}
      \Big]
    - \frac{\tau}{2}q_4M^{m_2\times m_1}v^{n-1}
    + \tau\mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}),
    \end{aligned}
    ```

!!! details " Jacobian matrix calculation"
    ```math
    JH(X) 
    = Q
    + 
    \begin{bmatrix}
    \Big[
      \frac{\tau}{2}\alpha^{n-\frac{1}{2}} JG\big(\frac{X_{1:m_1}+v^{n-1}}{2}\big)
    + \frac{\tau^2}{4} JF\big(\frac{\tau}{4}X_{1:m_1}+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
    \Big]^{m_1\times m_1}
    & 0^{m_1\times m_2}
    \\
    0^{m_2\times m_1} & 0^{m_2\times m_2}
    \end{bmatrix}
    ```