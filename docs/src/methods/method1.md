# Scheme 1
We first present a numerical scheme based on the Crank-Nicolson Galerkin method, which consists of finding ``U^n, V^n \in \mathcal{V}_{m_1}`` and ``Z^n, R^n \in \mathcal{V}_{m_2}`` such that
```math
\begin{align}
\label{def:approx1}
\begin{aligned}
& \big(\varphi,\bar{\partial}V^n\big)
+ \alpha(t_{n-\frac{1}{2}})\Big[
    \big(\nabla\varphi,\nabla\widehat{U}^n\big)
  - \big(\varphi,\widehat{R}^n)_{\Gamma_1}
  + \big(\varphi,g(\widehat{V}^n)\big)_{\Gamma_1}\Big]
+ \big(\varphi,f(\widehat{U}^n)\big) 
= \big(\varphi,f_1(t_{n-\frac{1}{2}}\big),
\quad\forall\varphi\in \mathcal{V}_{m_1},
\\
& \big(\phi,q_1\bar{\partial}R^n
+ q_2\widehat{R}^n
+ q_3\widehat{Z}^n
+ q_4\widehat{V}^n\big)_{\Gamma_1}
= \big(\phi,f_2(t_{n-\frac{1}{2}})\big)_{\Gamma_1},
\quad\forall\phi\in \mathcal{V}_{m_2},
\\
& 
\bar{\partial}U^n = \widehat{V}^n,\quad 
\bar{\partial}Z^n = \widehat{R}^n,
\end{aligned}
\end{align}
```
with ``U^0, V^0 \in \mathcal{V}_{m_1}`` and ``Z^0, R^0 \in \mathcal{V}_{m_2}`` given as approximations of the initial solutions ``u_0, v_0, z_0``, and ``r_0``.

!!! details "Notation"
    - ``\mathcal{V}_{m_1} \subset H_{\Gamma_0}^1(\Omega)``: Subspace of dimension ``m_1`` with basis ``\{\varphi_j\}_{j=1}^{m_1}``.
    - ``\mathcal{V}_{m_2} = \mathcal{V}_{m_1}|_{\Gamma_1}``: Subspace of dimension ``m_2`` with basis ``\{\phi_j\}_{j=1}^{m_2}``.
    - ``\displaystyle w^n:=w(t_n),\quad \bar{\partial}w^n := \frac{w^n - w^{n-1}}{\tau}\approx w^\prime(t_{n-\frac{1}{2}}),\quad \widehat{w}^n := \frac{w^n + w^{n-1}}{2}\approx w(t_{n-\frac{1}{2}}),``
    where ``\tau`` denotes the time step, ``t_n = n\tau`` the discrete times, ``t_{n-\frac{1}{2}}`` the midpoint of ``[t_{n-1},t_{n}]``, and ``w`` an arbitrary time-dependent function.
    
## Matrix formulation
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
+ \alpha(t_{n-\frac{1}{2}})\Big[  
    K^{m_1\times m_1}\widehat{d}^n
  - M^{m_1\times m_2}\widehat{r}^n
  + G^{m_1}(\widehat{v}^n)\Big]
+ F^{m_1}(\widehat{d}^n)
= \mathcal{F}_1^{m_1}(t_{n-\frac{1}{2}}),
\\
& M^{m_2\times m_2}\big[
  q_1\bar{\partial}r^n
+ q_2\widehat{r}^n
+ q_3\widehat{z}^n]
+ q_4M^{m_2\times m_1}\widehat{v}^n
= \mathcal{F}_2^{m_2}(t_{n-\frac{1}{2}}),
\\
& \bar{\partial}d^n = \widehat{v}^n,\quad
\bar{\partial}z^n = \widehat{r}^n.
\end{aligned}
\end{align}
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
    [\mathcal{F}_1^{m_1}(t)]_i = \big(\varphi_i,f_1(t)\big),\quad
    [\mathcal{F}_2^{m_2}(t)]_i = \big(\phi_i,f_2(t)\big)_{\Gamma_1},
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

## Nonlinear solver

Using ``d^n = d^{n-1} + \frac{\tau}{2}(v^n+v^{n-1})`` and ``z^n = z^{n-1} + \frac{\tau}{2}(r^n+r^{n-1})`` into system \eqref{def:approx1:mat_form} yields
```math
\label{prob1:nonlinear_system}
\begin{aligned}
Q^{(m_1+m_2)\times(m_1+m_2)}
\begin{bmatrix}
 v^n \\[5pt]
 r^n
\end{bmatrix}
+ 
\begin{bmatrix}
 \tau\alpha(t_{n-\frac{1}{2}})
 G^{m_1}\big(\frac{v^n+v^{n-1}}{2}\big) 
 + \tau F^{m_1}\big(\frac{\tau}{4}v^{n}+\frac{\tau}{4}v^{n-1}+d^{n-1}\big) \\[5pt]
 0^{m_2}
\end{bmatrix}
-
\begin{bmatrix}
L^{m_1}\\
L^{m_2}
\end{bmatrix}
= 0,
\end{aligned}
```
which can be reformulated as the problem of finding ``X\in\mathbb{R}^{m_1+m_2}``, where ``X = \begin{bmatrix}v^n\\r^n\end{bmatrix}``, such that
```math
H(X)=0.
```

We employ Newton's method to solve this nonlinear system. Taking ``X_0=[v^{n-1};r^{n-1}]`` as the initial guess, Newton's method generates the sequence ``X_{\kappa+1}= X_\kappa + S_\kappa``, where ``S_\kappa`` is obtained by solving the linear system
```math
JH(X_\kappa) S_\kappa = -H(X_\kappa),
```
and ``JH(X_\kappa)`` denotes the Jacobian matrix of ``H`` evaluated at ``X_\kappa``.

!!! details " Jacobian matrix calculation"
    Initially, note that:
    ```math
    H_i(X) = \sum_{\ell=1}^{m_1+m_2}Q_{i,\ell}X_\ell
    + 
    \begin{cases}
    \tau\alpha(t_{n-\frac{1}{2}})
    G_i^{m_1}\big(\frac{X[1:m_1]+v^{n-1}}{2}\big) 
    + \tau F_i^{m_1}\big(\frac{\tau}{4}X[1:m_1]+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
    - L_i^{m_1} 
    & \text{if } i\in\{1, \ldots, m_1\}
    \\
    -L_i^{m_2} 
    & \text{if } i\in\{m_1+1, \ldots, m_1+m_2\}
    \end{cases}
    ```
    In this way,
    ```math
    \frac{\partial H_i}{\partial X_j}(X) 
    = Q_{i,j} 
    +
    \begin{cases}
    \tau\alpha(t_{n-\frac{1}{2}})
    \frac{\partial}{\partial X_j}G_i^{m_1}\big(\frac{X[1:m_1]+v^{n-1}}{2}\big) 
    + \tau \frac{\partial}{\partial X_j}F_i^{m_1}\big(\frac{\tau}{4}X[1:m_1]+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
    & \text{if } i\in\{1, \ldots, m_1\}
    \\
    0
    & \text{if } i\in\{m_1+1, \ldots, m_1+m_2\}
    \end{cases}
    ```
    where
    ```math
    G_i^{m_1}(v) 
    = \int_{\Gamma_1}\varphi_i(x)g\big(x,\sum_{\ell=1}^{m_1}v_\ell\varphi_\ell(x)\big)d\Gamma
    \;
    \Rightarrow
    \;
    \frac{\partial}{\partial X_j} G_i^{m_1}\big(\frac{X[1:m_1]+v^{n-1}}{2}\big)
    = \frac{1}{2}\int_{\Gamma_1}\varphi_i(x)\varphi_j(x)\frac{\partial g}{\partial s}\big(x,\sum_{\ell=1}^{m_1}\frac{X_\ell+v_{\ell}^{n-1}}{2}\varphi_\ell(x)\big)d\Gamma
    ```
    and
    ```math
    F_i^{m_1}(d) 
    = 
    \int_{\Omega}\varphi_i(x)f\Big(\sum_{\ell=1}^{m_1}d_\ell\varphi_\ell(x)\Big)dx
    \;
    \Rightarrow
    \;
    \frac{\partial}{\partial X_j}F_i^{m_1}\big(\frac{\tau}{4}X[1:m_1]+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
    = 
    \frac{\tau}{4}\int_{\Omega}\varphi_i(x)\varphi_j(x)f^\prime\Big(\sum_{\ell=1}^{m_1}
    \big[\frac{\tau}{4}X_\ell+\frac{\tau}{4}v_\ell^{n-1}+d_\ell^{n-1}\big]
    \varphi_\ell(x)\Big)dx.
    ```
    Consequently, the Jacobian matrix can be expressed as:
    ```math
    JH(X) 
    = Q
    + 
    \begin{bmatrix}
    \Big[
      \frac{\tau}{2}\alpha(t_{n-\frac{1}{2}}) JG\big(\frac{X[1:m_1]+v^{n-1}}{2}\big)
    + \frac{\tau^2}{4} JF\big(\frac{\tau}{4}X[1:m_1]+\frac{\tau}{4}v^{n-1}+d^{n-1}\big)
    \Big]^{m_1\times m_1}
    & 0^{m_1\times m_2}
    \\
    0^{m_2\times m_1} & 0^{m_2\times m_2}
    \end{bmatrix}
    ```

!!! details "Matrix and vector definitions"
    ```math
    \begin{aligned}
    &Q^{(m_1+m_2)\times(m_1+m_2)} =
    \begin{bmatrix}
      M^{m_1\times m_1} + \frac{\tau^2\alpha(t_{n-\frac{1}{2}})}{4}K^{m_1\times m_1} 
    &-\frac{\tau\alpha(t_{n-\frac{1}{2}})}{2}M^{m_1\times m_2}  
    \\[5pt]
      \frac{\tau}{2}q_4M^{m_2\times m_1}
    & (q_1 + \frac{\tau}{2}q_2 + \frac{\tau^2}{4}q_3)M^{m_2\times m_2}
    \end{bmatrix},
    \\[10pt]
    &L^{m_1} = 
    M^{m_1\times m_1}v^{n-1}
    - \tau\alpha(t_{n-\frac{1}{2}}) K^{m_1\times m_1}\Big(\frac{\tau}{4}v^{n-1}+d^{n-1}\Big)
    + \frac{\tau}{2}\alpha(t_{n-\frac{1}{2}})M^{m_1\times m_2}r^{n-1}
    + \tau\mathcal{F}_1^{m_1}(t_{n-\frac{1}{2}}),
    \\[10pt]
    &L^{m_2} =
    M^{m_2\times m_2}\Big[
      (q_1 - \frac{\tau}{2}q_2 - \frac{\tau^2}{4}q_3)r^{n-1}
      - \tau q_3 z^{n-1}
      \Big]
    - \frac{\tau}{2}q_4M^{m_2\times m_1}v^{n-1}
    + \tau\mathcal{F}_2^{m_2}(t_{n-\frac{1}{2}}),
    \\[10pt]
    & JG_{i,j}(v) 
    = 
    \int_{\Gamma_1}\varphi_i(x)\varphi_j(x)\frac{\partial g}{\partial s}\big(x,\sum_{\ell=1}^{m_1}v_\ell\varphi_\ell(x)\big)d\Gamma,
    \\[5pt]
    & JF_{i,j}(d)
    =
    \int_{\Omega}\varphi_i(x)\varphi_j(x)f^\prime\Big(\sum_{\ell=1}^{m_1}d_\ell\varphi_\ell(x)\Big)dx.
    \end{aligned}
    ```