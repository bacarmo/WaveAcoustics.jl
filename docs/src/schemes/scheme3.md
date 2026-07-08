# Scheme 3
The following scheme is defined using the linearized Crank-Nicolson Galerkin method, which consists of finding ``U^n, V^n \in \mathcal{V}_{m_1}`` and ``Z^n, R^n \in \mathcal{V}_{m_2}`` such that
```math
\begin{align*}
\begin{aligned}
& \big(\varphi,\bar{\partial}V^n\big)
+ \alpha^{n-\frac{1}{2}}\Big[
    \big(\nabla\varphi,\nabla\widehat{U}^n\big)
  - \big(\varphi,\widehat{R}^n)_{\Gamma_1}
  + \big(\varphi,g(V^{\ast n})\big)_{\Gamma_1}\Big]
+ \big(\varphi,f(U^{\ast n})\big) 
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
\end{align*}
```
for ``n = \text{“1,0''},\,1,\,2,\,\ldots``, with ``U^0, V^0 \in \mathcal{V}_{m_1}`` and ``Z^0, R^0 \in \mathcal{V}_{m_2}`` given as approximations of the initial conditions ``u_0, v_0, z_0``, and ``r_0``, defined in the same way as in Scheme 1.

## Matrix Formulation
```math
\begin{align*}
\begin{aligned}
& M^{m_1\times m_1}\bar{\partial}v^n
+ \alpha^{n-\frac{1}{2}}\Big[  
    K^{m_1\times m_1}\widehat{d}^n
  - M^{m_1\times m_2}\widehat{r}^n
  + G^{m_1}(v^{\ast n})\Big]
+ F^{m_1}(d^{\ast n})
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
\bar{\partial}z^n = \widehat{r}^n.
\end{aligned}
\end{align*}
```

## Solving the Algebraic Systems
```math
Q(n) \widehat{v}^n 
= L(n,v^{*n},d^{*n}).
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
and the remaining quantities ``d^n``, ``v^n``, ``z^n``, and ``r^n`` by
```math
\begin{align*}
& v^n = 2\hat{v}^n - v^{n-1},
\quad r^n = 2\hat{r}^n - r^{n-1},
\quad d^n = \tau\hat{v}^n + d^{n-1},
\quad z^n = \tau\hat{r}^n + z^{n-1}.
\end{align*}
```

!!! details "Matrix and vector definitions"
    ```math
    \begin{align*}
    Q = & 
    2M^{m_1\times m_1} 
    + \frac{\tau^2}{2}\alpha^{n-\frac{1}{2}}K^{m_1\times m_1}
    + \frac{\tau^2q_4}{q_5}\alpha^{n-\frac{1}{2}}
    \begin{bmatrix}
    M^{m_2\times m_2}       & 0^{m_2\times(m_1-m_2)}\\[5pt]
    0^{(m_1-m_2)\times m_2} & 0^{(m_1-m_2)\times(m_1-m_2)}
    \end{bmatrix}
    \\[10pt]
    L =&
    -\tau\alpha^{n-\frac{1}{2}} G^{m_1}(v^{\ast n})  
    - \tau F^{m_1}(d^{\ast n})
    + 2M^{m_1\times m_1}v^{n-1}
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
