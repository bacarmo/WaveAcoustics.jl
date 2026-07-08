# Scheme 1
The numerical scheme present is based on the Crank-Nicolson Galerkin method, which consists of finding ``U^n, V^n \in \mathcal{V}_{m_1}``, and ``Z^n, R^n \in \mathcal{V}_{m_2}`` such that
```math
\begin{align*}
& \big(\varphi,\bar{\partial}V^n\big)
+ \alpha^{n-\frac{1}{2}}\Big[
    \big(\nabla\varphi,\nabla\widehat{U}^n\big)
  - \big(\varphi,\widehat{R}^n)_{\Gamma_1}
  + \big(\varphi,g(\widehat{V}^n)\big)_{\Gamma_1}\Big]
+ \big(\varphi,f(\widehat{U}^n)\big) 
= \big(\varphi,f_1^{n-\frac{1}{2}}\big),
\quad\forall\varphi\in \mathcal{V}_{m_1},
\\[5pt]
& \big(\phi,q_1\bar{\partial}R^n
+ q_2\widehat{R}^n
+ q_3\widehat{Z}^n
+ q_4\widehat{V}^n\big)_{\Gamma_1}
= \big(\phi,f_2^{n-\frac{1}{2}}\big)_{\Gamma_1},
\quad\forall\phi\in \mathcal{V}_{m_2},
\\[5pt]
& 
\bar{\partial}U^n = \widehat{V}^n,\quad 
\bar{\partial}Z^n = \widehat{R}^n,
\end{align*}
```
with ``U^0, V^0 \in \mathcal{V}_{m_1}``, and ``Z^0, R^0 \in \mathcal{V}_{m_2}`` given as approximations of the initial solutions ``u_0``, ``\, v_0``, ``\, z_0``, and ``r_0``.
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
    - ``\mathcal{V}_{m_2} = \mathcal{V}_{m_1}|_{\Gamma_1}``: Subspace of dimension ``m_2`` with basis ``\{\phi_j\}_{j=1}^{m_2}``, where the basis functions of ``\mathcal{V}_{m_1}``
      are indexed so that
    ```math
        \phi_j = \varphi_j|_{\Gamma_1}, \quad j = 1, \dots, m_2.
    ```
    - For a sufficiently regular time-dependent function ``w``, ``\tau`` the time step, and
      ``t_n = n\tau``:
      ```math
          w^n := w(t_n), \qquad
          \bar{\partial}w^n := \frac{w^n - w^{n-1}}{\tau} \approx w^\prime(t_{n-\frac{1}{2}}), \qquad
          \widehat{w}^n := \frac{w^n + w^{n-1}}{2} \approx w(t_{n-\frac{1}{2}}),
      ```
      where ``t_{n-\frac{1}{2}}`` denotes the midpoint of ``[t_{n-1}, t_n]``.

## Matrix Formulation
Representing the approximate solutions in terms of the basis functions,
```math
U^n      = \sum_{j=1}^{m_1} d_j^n\varphi_j,\;
V^n      = \sum_{j=1}^{m_1} v_j^n\varphi_j,\;
Z^n      = \sum_{j=1}^{m_2} z_j^n\phi_j,\;
R^n      = \sum_{j=1}^{m_2} r_j^n\phi_j,
```
and choosing test functions ``\varphi=\varphi_i`` for ``i = 1, \ldots, m_1``, and ``\phi=\phi_i`` for ``i = 1, \ldots, m_2``, we obtain the system
```math
\begin{align*}
& M^{m_1\times m_1}\bar{\partial}v^n
+ \alpha^{n-\frac{1}{2}}\Big[  
    K^{m_1\times m_1}\widehat{d}^n
  - M^{m_1\times m_2}\widehat{r}^n
  + G^{m_1}(\widehat{v}^n)\Big]
+ F^{m_1}(\widehat{d}^n)
= \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}),
\\[3pt]
& M^{m_2\times m_2}\big[
  q_1\bar{\partial}r^n
+ q_2\widehat{r}^n
+ q_3\widehat{z}^n\big]
+ q_4M^{m_2\times m_1}\widehat{v}^n
= \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}),
\\[5pt]
& \bar{\partial}d^n = \widehat{v}^n,\quad
\bar{\partial}z^n = \widehat{r}^n,
\end{align*}
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
    \mathcal{F}_i^{m_1}(w) = \big(\varphi_i,w\big),\quad
    \mathcal{F}_i^{m_2}(w) = \big(\phi_i,w\big)_{\Gamma_1},\quad
    \kappa^{m_1}(w) = \big(\nabla\varphi_i,\nabla w\big),
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
    !!! note "Beware!"
        - Although ``q_1``, ``q_2``, and ``q_3`` are functions in the original model, we treat
          them as constants from this point on. This is a deliberate simplification adopted to reduce the complexity of the numerical implementation.
        - The index convention imposed on the basis functions of ``\mathcal{V}_{m_1}`` and ``\mathcal{V}_{m_2}``, namely
          ```math
                \phi_j = \varphi_j|_{\Gamma_1}, \quad j = 1, \dots, m_2,
          ```
          allows us to rewrite the matrices ``M^{m_1\times m_2}`` and ``M^{m_2\times m_1}`` as
          ```math
                M^{m_1\times m_2} = \begin{bmatrix} M^{m_2\times m_2} \\ 0^{(m_1-m_2)\times m_2} \end{bmatrix}
                \quad\text{and}\quad
                M^{m_2\times m_1} = \begin{bmatrix} M^{m_2\times m_2} & 0^{m_2\times(m_1-m_2)} \end{bmatrix}.
          ```
          Furthermore, the nonlinear term ``G^{m_1}(\widehat{v}^n) `` can be rewritten as
          ```math
          G^{m_1}(\widehat{v}^n) 
          = \begin{bmatrix} G^{m_2}(\widehat{v}_{1:m_2}^n)\\ 0^{(m_1-m_2)} \end{bmatrix},
          ```
          with
          ```math
          G^{m_2}_i(\widehat{v}_{1:m_2}^n) 
          =\int_{\Gamma_1}\phi_i(x)g\Big(x,\sum_{\ell=1}^{m_2}\widehat{v}_\ell^n\phi_\ell(x)\Big)d\Gamma.
          ```

## Solving the Nonlinear Algebraic System
In what follows, we show that the matrix formulation is equivalent to finding
``X \in \mathbb{R}^{m_1}`` such that ``H(X) = 0``, 
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

### Formulation in Term of ``X=\hat{v}^n``
```math
Q(n) \hat{v}^n
+ \tau\alpha^{n-\frac{1}{2}} G^{m_1}(\widehat{v}^n) + \tau F^{m_1}(\frac{\tau}{2}\widehat{v}^n + d^{n-1})
- L_1(n)
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
  \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}).
```

The remaining quantities ``d^n``, ``v^n``, ``z^n``, and ``r^n`` are then recovered by
```math
\begin{align*}
& v^n = 2\hat{v}^n - v^{n-1},
\quad r^n = 2\hat{r}^n - r^{n-1},
\quad d^n = \tau\hat{v}^n + d^{n-1},
\quad z^n = \tau\hat{r}^n + z^{n-1}.
\end{align*}
```

!!! details "Details"
    To rewrite the matrix formulation in terms of ``\widehat{v}^n`` and ``\widehat{r}^n``, we make use of the following identities:
    ```math
    \bar\partial v^n = \frac{2}{\tau}\widehat{v}^n - \frac{2}{\tau}v^{n-1},
    \quad
    \widehat{d}^n = \frac{\tau}{2} \widehat{v}^n + d^{n-1},
    \quad
    \bar\partial r^n = \frac{2}{\tau}\widehat{r}^n - \frac{2}{\tau}z^{n-1},
    \quad
    \widehat{z}^n = \frac{\tau}{2} \widehat{r}^n + z^{n-1}.
    ```
    Applying these identities into the matrix formulation, we obtain:
    ```math
    \begin{align*}
    & M^{m_1\times m_1} \big(\frac{2}{\tau}\widehat{v}^n-\frac{2}{\tau}v^{n-1}\big)
    + \alpha^{n-\frac{1}{2}}\Big[  
        K^{m_1\times m_1} \big(\frac{\tau}{2}\widehat{v}^n+d^{n-1}\big)
      - M^{m_1\times m_2}\widehat{r}^n
      + G^{m_1}(\widehat{v}^n)\Big]
    + F^{m_1}(\frac{\tau}{2}\widehat{v}^n + d^{n-1})
    = \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}),
    \\[10pt]
    & M^{m_2\times m_2}\big[
      q_1\big(\frac{2}{\tau}\widehat{r}^n-\frac{2}{\tau}r^{n-1}\big)
    + q_2\widehat{r}^n
    + q_3\big(\frac{\tau}{2}\widehat{r}^n+z^{n-1}\big)\big]
    + q_4M^{m_2\times m_1}\widehat{v}^n
    = \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}).
    \end{align*}
    ```
    Isolating ``\widehat{r}^n`` in the second equation:
    ```math
    (2q_1+\tau q_2+\frac{\tau^2}{2}q_3)M^{m_2\times m_2} \widehat{r}^n
    =
    - \tau q_4M^{m_2\times m_2} \widehat{v}_{1:m_2}^n
    + M^{m_2\times m_2}\big(2q_1r^{n-1} - \tau q_3z^{n-1}\big)
    + \tau \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}).
    ```
    Denoting ``q_5 = 2q_1+\tau q_2+\frac{\tau^2}{2}q_3`` and using the result above in the first equation, we obtain:
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
    !!! note "Beware!"
        ```math
        \begin{align*}
        &
        \bar\partial v^n 
        = \frac{v^n-v^{n-1}}{\tau}
        = \frac{2\frac{v^n+v^{n-1}}{2}-v^{n-1}-v^{n-1}}{\tau}
        = \frac{2\widehat{v}^n-2v^{n-1}}{\tau}
        \quad\Rightarrow\quad
        \bar\partial v^n = \frac{2}{\tau}\widehat{v}^n - \frac{2}{\tau}v^{n-1}
        \\[10pt]
        &
        \bar\partial d^n = \widehat{v}^n
        \;\;\Rightarrow\;\;
        \frac{d^n-d^{n-1}}{\tau} = \widehat{v}^n
        \;\;\Rightarrow\;\;
        d^n = \tau \widehat{v}^n + d^{n-1}
        \;\;\Rightarrow\;\;
        \frac{d^n+d^{n-1}}{2} = \frac{\tau}{2} \widehat{v}^n + d^{n-1}
        \quad\Rightarrow\quad
        \widehat{d}^n = \frac{\tau}{2} \widehat{v}^n + d^{n-1}
        \end{align*}
        ```

!!! details "Matrix and vector definitions"
    ```math
    \begin{align*}
    Q =&
    2M^{m_1\times m_1} 
    + \frac{\tau^2}{2}\alpha^{n-\frac{1}{2}}K^{m_1\times m_1}
    + \frac{\tau^2q_4}{q_5}\alpha^{n-\frac{1}{2}}
    \begin{bmatrix}
    M^{m_2\times m_2}       & 0^{m_2\times(m_1-m_2)}
    \\[5pt]
    0^{(m_1-m_2)\times m_2} & 0^{(m_1-m_2)\times(m_1-m_2)}
    \end{bmatrix}
    \\[30pt]
    L =&
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
    where ``q_5 = 2q_1+\tau q_2+\frac{\tau^2}{2}q_3``.

!!! details " Jacobian matrix calculation"
    ```math
    JH(X) 
    = Q
    + \tau\alpha^{n-\frac{1}{2}} JG\big(X\big)
    + \frac{\tau^2}{2} JF\big(\frac{\tau}{2}X+d^{n-1}\big)
    ```
    !!! note "Beware!"
        ```math
        JG^{m_1\times m_1}(X) =
        \begin{bmatrix}\displaystyle
        JG^{m_2\times m_2}(X_{1:m_2}) & 0^{m_2\times(m_1-m_2)}
        \\[10pt]
        0^{(m_1-m_2)\times m_2}       & 0^{(m_1-m_2)\times(m_1-m_2)}
        \end{bmatrix}
        ```
        ```math
        JG^{m_2\times m_2}_{i,j}(X_{1:m_2})
        = \int_{\Gamma_1}\phi_i(x)\phi_j(x)\frac{\partial g}{\partial s}\big(x,\sum_{\ell=1}^{m_2}X_\ell\phi_\ell(x)\big)d\Gamma
        ```
        ```math
        JF_{i,j}(\frac{\tau}{2}X+d^{n-1}) = 
        \int_{\Omega}\varphi_i(x)\varphi_j(x)f^\prime\Big(\sum_{\ell=1}^{m_1}
        \big[\frac{\tau}{2}X_\ell+d_\ell^{n-1}\big]
        \varphi_\ell(x)\Big)dx
        ```