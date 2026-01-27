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