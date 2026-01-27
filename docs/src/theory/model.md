## Strong Formulation
We seek functions ``u`` and ``z`` satisfying the following system of equations:
```math
\begin{align}\label{pde:mdl}
\begin{aligned}
& u^{\prime\prime}(x,t) 
- \alpha(t)\Delta u(x,t) 
+ f\big(u(x,t)\big) 
= f_1(x,t),
\quad(x,t)\in\Omega\times(0,+\infty),
\\[5pt]
& q_1 z^{\prime\prime}(x,t)
+ q_2 z^\prime(x,t)
+ q_3 z(x,t)
+ q_4 u^\prime(x,t)
= f_2(x,t),
\quad(x,t)\in\Gamma_1\times(0,+\infty),
\\[5pt]
& \frac{\partial u}{\partial\nu}(x,t)
= z^\prime(x,t)
- g\big(x,u^\prime(x,t)\big),
\quad(x,t)\in\Gamma_1\times(0,+\infty),
\\[5pt]
& u(x,t) = 0,\quad (x,t)\in\Gamma_0\times(0,+\infty),
\end{aligned}
\end{align}
```
with initial conditions
```math
\begin{align}\label{pde:mdl:initial_condition}
\begin{aligned}
& u(x,0) = u_0(x),\quad u^\prime(x,0)=v_0(x),\quad x\in\Omega,
\\
& z(x,0) = z_0(x),\quad
  z^\prime(x,0) = r_0(x) \equiv \frac{\partial u_0}{\partial\nu}(x) + g\big(x,v_0(x)\big),\quad x\in\Gamma_1,
\end{aligned}
\end{align}
```
where ``\Omega`` is a bounded open subset of ``\mathbb{R}^n``, ``n\geq 2``, with smooth boundary ``\Gamma=\Gamma_0\cup\Gamma_1`` and disjoint ``\Gamma_0``, ``\Gamma_1``.


Existence and uniqueness results for particular cases of \eqref{pde:mdl}-\eqref{pde:mdl:initial_condition} can be found in Alcântara et al. (2025) "Numerical analysis for nonlinear wave equations with boundary conditions: Dirichlet, Acoustics and Impenetrability", *Applied Mathematics and Computation*, 484, 129009, [https://doi.org/10.1016/j.amc.2024.129009](https://doi.org/10.1016/j.amc.2024.129009).

## Weak Formulation
We seek functions $u(t)\in H_{\Gamma_0}^1(\Omega)$ and $z(t)\in L^2(\Gamma_1)$ such that
```math
\begin{align}
\label{pde:variational_form}
\begin{aligned}
& \big(\varphi,u^{\prime\prime}(t)\big)
+ \alpha(t)\Big[
    \big(\nabla\varphi,\nabla u(t)\big)
  - \big(\varphi,z^\prime(t))_{\Gamma_1}
  + \big(\varphi,g\big(u^\prime(t)\big)\big)_{\Gamma_1}\Big]
+ \big(\varphi,f\big(u(t)\big)\big) 
= \big(\varphi,f_1(t)\big),
\quad\forall\varphi\in H_{\Gamma_0}^1(\Omega),
\\
& \big(\phi,q_1z^{\prime\prime}(t)
+ q_2z^\prime(t)
+ q_3z(t)
+ q_4u^\prime(t)\big)_{\Gamma_1}
= \big(\phi,f_2(t)\big)_{\Gamma_1},
\quad\forall\phi\in L^2(\Gamma_1),
\end{aligned}
\end{align}
```
with 
$u(0)=u_0$, 
$u^\prime(0)=v_0$, 
$z(0)=z_0$, and 
$z^\prime(0) = r_0 \equiv \frac{\partial u_0}{\partial\nu} + g(v_0)$. 


We consider 
``H_{\Gamma_0}^1(\Omega) = \{ v \in H^1(\Omega);\, v|_{\Gamma_0} = 0 \}`` 
and the inner products and norms in ``L^2(\Omega)`` and ``L^2(\Gamma_1)`` by
```math
(\cdot, \cdot),\quad
(\cdot, \cdot)_{\Gamma_1},\quad
\|\cdot\|,\quad
\|\cdot\|_{\Gamma_1}.
```

By introducing the auxiliary variables ``v(t)=u^\prime(t)`` and ``r(t)=z^\prime(t)``, we obtain the equivalent first-order system: find functions ``u(t),v(t)\in H_{\Gamma_0}^1(\Omega)`` and ``z(t),r(t)\in L^2(\Gamma_1)`` such that
```math
\begin{align}
\label{pde:variational_form_opt2}
\begin{aligned}
& \big(\varphi,v^{\prime}(t)\big)
+ \alpha(t)\Big[
    \big(\nabla\varphi,\nabla u(t)\big)
  - \big(\varphi,r(t)\big)_{\Gamma_1}
  + \big(\varphi,g\big(v(t)\big)\big)_{\Gamma_1}\Big]
+ \big(\varphi,f\big(u(t)\big)\big) 
= \big(\varphi,f_1(t)\big),
\quad\forall\varphi\in H_{\Gamma_0}^1(\Omega),
\\
& \big(\phi,q_1r^{\prime}(t)
+ q_2r(t)
+ q_3z(t)
+ q_4v(t)\big)_{\Gamma_1}
= \big(\phi,f_2(t)\big)_{\Gamma_1},
\quad\forall\phi\in L^2(\Gamma_1),
\\
& u^\prime(t)=v(t),\quad z^\prime(t)=r(t),
\end{aligned}
\end{align}
```
with initial conditions
``u(0)=u_0``, 
``v(0)=v_0``, 
``z(0)=z_0``, and 
``r(0) = r_0 \equiv \frac{\partial u_0}{\partial\nu} + g(v_0)``.
