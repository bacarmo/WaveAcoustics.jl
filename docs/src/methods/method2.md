# Scheme 2
The second scheme is defined using the linearized Crank-Nicolson Galerkin method, which consists of finding ``U^n, V^n \in \mathcal{V}_{m_1}`` and ``Z^n, R^n \in \mathcal{V}_{m_2}`` such that
```math
\begin{align}
\label{def:approx2}
\begin{aligned}
& \big(\varphi,\bar{\partial}V^n\big)
+ \alpha(t_{n-\frac{1}{2}})\Big[
    \big(\nabla\varphi,\nabla\widehat{U}^n\big)
  - \big(\varphi,\widehat{R}^n)_{\Gamma_1}
  + \big(\varphi,g(V^{\ast n})\big)_{\Gamma_1}\Big]
+ \big(\varphi,f(U^{\ast n})\big) 
= \big(\varphi,f_1(t_{n-\frac{1}{2}}\big),
\quad\forall\varphi\in \mathcal{V}_{m_1},
\\
& \big(\phi,q_1\bar{\partial}R^n
+ q_2\widehat{R}^n
+ q_3\widehat{Z}^n
+ q_4V^{\ast n}\big)_{\Gamma_1}
= \big(\phi,f_2(t_{n-\frac{1}{2}})\big)_{\Gamma_1},
\quad\forall\phi\in \mathcal{V}_{m_2},
\\
& 
\bar{\partial}U^n = \widehat{V}^n,\quad 
\bar{\partial}Z^n = \widehat{R}^n,
\end{aligned}
\end{align}
```
for ``n = \text{“1,0''},\,1,\,2,\,\ldots``, with ``U^0, V^0 \in \mathcal{V}_{m_1}`` and ``Z^0, R^0 \in \mathcal{V}_{m_2}`` given as approximations of the initial solutions ``u_0, v_0, z_0``, and ``r_0``.

!!! details "Note"
    - The first two time steps in \eqref{def:approx2} constitute a single-step predictor-corrector initialization.
    - In the prediction step (case ``n=\text{“1,0''}``), temporary approximations ``U^{\text{“1,0''}}``, ``V^{\text{“1,0''}}``, ``Z^{\text{“1,0''}}``, and ``R^{\text{“1,0''}}`` at ``t_1`` are computed using the initial solution ``U^0``, ``V^0``, ``Z^0``, and ``R^0``.    
    - In the correction step (case ``n=1``), the temporary approximations from the prediction step, together with the initial solutions, are used to obtain the definitive approximations ``U^1``, ``V^1``, ``Z^1``, and ``R^1``.
    - For ``n \geq 2``,  the approximations are generated using information from the two previous time steps at the nonlinear terms and the coupling term in the acoustic equation.
    - At each time step, two independent linear systems are solved, as a result of choices that lead to decoupling the second equation and linearization of the nonlinear terms.

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