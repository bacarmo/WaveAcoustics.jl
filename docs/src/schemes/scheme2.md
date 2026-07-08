# Scheme 2
The following fully discrete scheme combines the Crank-Nicolson Galerkin method with selective linearization of nonlinear terms. 
It consists of finding ``U^n, V^n \in \mathcal{V}_{m_1}``, and ``Z^n, R^n \in \mathcal{V}_{m_2}`` such that
```math
\begin{align*}
& \big(\varphi,\bar{\partial}V^n\big)
+ \alpha^{n-\frac{1}{2}}\Big[
    \big(\nabla\varphi,\nabla\widehat{U}^n\big)
  - \big(\varphi,\widehat{R}^n)_{\Gamma_1}
  + \big(\varphi,g(\widehat{V}^n)\big)_{\Gamma_1}\Big]
+ \big(\varphi,f(U^{\ast n})\big) 
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
for ``n = \text{“1,0''},\,1,\,2,\,\ldots``, with ``U^0, V^0 \in \mathcal{V}_{m_1}``, and ``Z^0, R^0 \in \mathcal{V}_{m_2}`` given as approximations of the initial solutions ``u_0``, ``\, v_0``, ``\, z_0``, and ``r_0``, defined in the same manner as in Scheme 1.

!!! details "Notation"
    In addition to the operators 
    ``\displaystyle\bar{\partial}w^n=\frac{w^n - w^{n-1}}{\tau}`` and 
    ``\displaystyle\widehat{w}^n = \frac{w^n + w^{n-1}}{2}``, consider
    ```math
    \bar{\partial}w^{\text{“1,0''}} = \frac{w^{\text{“1,0''}} - w^0}{\tau},\quad
    \widehat{w}^{\text{“1,0''}} = \frac{w^{\text{“1,0''}} + w^0}{2},
    \quad\text{and}\quad
    w^{*n} = 
    \begin{cases}\displaystyle
    w^0,                             & \text{if } n = \text{“1,0''},
    \\[10pt] \displaystyle
    \frac{w^{\text{“1,0''}}+w^0}{2}, & \text{if } n = 1,
    \\[10pt] \displaystyle
    \frac{3w^{n-1}-w^{n-2}}{2},      & \text{if } n \geq 2.
    \end{cases}
    ```

## Matrix Formulation
```math
\begin{align*}
& M^{m_1\times m_1}\bar{\partial}v^n
+ \alpha^{n-\frac{1}{2}}\Big[  
    K^{m_1\times m_1}\widehat{d}^n
  - M^{m_1\times m_2}\widehat{r}^n
  + G^{m_1}(\widehat{v}^n)\Big]
+ F^{m_1}(d^{\ast n})
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
\bar{\partial}z^n = \widehat{r}^n.
\end{align*}
```
## Solving the Algebraic Systems
The matrix formulation can be rewritten as:
```math
\begin{align*}
&
\big[Q(n)\big]^{m_1\times m_1} \widehat{v}^n
+ \tau\alpha^{n-\frac{1}{2}} G^{m_1}(\widehat{v}^n) 
- L(n,d^{\ast n}) = 0,
\\[10pt]
& \hat{r}^n
=
- \frac{\tau q_4}{q_5}\hat{v}_{1:m_2}^n
+ \frac{2q_1}{q_5} r^{n-1}
- \frac{\tau q_3}{q_5} z^{n-1}
+ \frac{\tau}{q_5} \Big(M^{m_2\times m_2}\Big)^{-1}
  \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}).
\end{align*}
```
The first system is nonlinear in ``\widehat{v}^n``, independent of ``\widehat{r}^n``, and is solved via Newton's method.
Once ``\widehat{v}^n`` is available, ``\widehat{r}^n`` follow explicitly from the second equation.

With ``\hat{v}^n`` and ``\hat{r}^n`` determined, the remaining unknowns are updated via
```math
\begin{align*}
& v^n = 2\hat{v}^n - v^{n-1},
\quad r^n = 2\hat{r}^n - r^{n-1},
\quad d^n = \tau\hat{v}^n + d^{n-1},
\quad z^n = \tau\hat{r}^n + z^{n-1}.
\end{align*}
```

!!! details "Details"
    Rewriting the matrix formulation in terms of ``\widehat{v}^n`` and ``\widehat{r}^n``, we obtain:
    ```math
    \begin{align*}
    & M^{m_1\times m_1} \big(\frac{2}{\tau}\widehat{v}^n-\frac{2}{\tau}v^{n-1}\big)
    + \alpha^{n-\frac{1}{2}}\Big[  
        K^{m_1\times m_1} \big(\frac{\tau}{2}\widehat{v}^n+d^{n-1}\big)
      - M^{m_1\times m_2}\widehat{r}^n
      + G^{m_1}(\widehat{v}^n)\Big]
    + F^{m_1}(d^{\ast n})
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
    + F^{m_1}(d^{\ast n})
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
    = \mathcal{F}^{m_1}(f_1^{n-\frac{1}{2}}),
    \\[10pt]
    & \hat{r}^n
    =
    - \frac{\tau q_4}{q_5}\hat{v}_{1:m_2}^n
    + \frac{2q_1}{q_5} r^{n-1}
    - \frac{\tau q_3}{q_5} z^{n-1}
    + \frac{\tau}{q_5} \Big(M^{m_2\times m_2}\Big)^{-1}
      \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}).
    \end{align*}
    ```
    Isolating ``\widehat{v}^n``:
    ```math
    \begin{align*}
    &
    Q(n) \widehat{v}^n
    + \tau\alpha^{n-\frac{1}{2}} G^{m_1}(\widehat{v}^n) 
    - L(n,d^{\ast n}) = 0,
    \\[10pt]
    & \hat{r}^n
    =
    - \frac{\tau q_4}{q_5}\hat{v}_{1:m_2}^n
    + \frac{2q_1}{q_5} r^{n-1}
    - \frac{\tau q_3}{q_5} z^{n-1}
    + \frac{\tau}{q_5} \Big(M^{m_2\times m_2}\Big)^{-1}
      \mathcal{F}^{m_2}(f_2^{n-\frac{1}{2}}).
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
    M^{m_2\times m_2}       & 0^{m_2\times(m_1-m_2)}\\[5pt]
    0^{(m_1-m_2)\times m_2} & 0^{(m_1-m_2)\times(m_1-m_2)}
    \end{bmatrix}
    \\[10pt]
    L =&
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

!!! details " Jacobian matrix calculation"
    ```math
    JH(X) 
    = Q + \tau\alpha^{n-\frac{1}{2}}
      \begin{bmatrix}\displaystyle
      JG^{m_2\times m_2}(X_{1:m_2}) & 0^{m_2\times(m_1-m_2)}
      \\[10pt]
      0^{(m_1-m_2)\times m_2}       & 0^{(m_1-m_2)\times(m_1-m_2)}
      \end{bmatrix} ,
    ```
    where
    ```math
    JG_{i,j}^{m_2\times m_2}(X_{1:m_2})
    =
    \int_{\Gamma_1} \phi_i(x)\phi_j(x)
    \frac{\partial g}{\partial s}\Big(x,\sum_{\ell=1}^{m_2}X_\ell\phi_\ell(x)\Big)d\Gamma.
    ```