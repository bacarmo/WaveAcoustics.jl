# Scheme 2
```julia-repl
julia> using WaveAcoustics

julia> cases1a = (
    (fe = Lagrange{1}, id = example1_manufactured(1.76), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{1}, id = example1_manufactured(2.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{2}, id = example1_manufactured(2.58), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{2}, id = example1_manufactured(3.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{3}, id = example1_manufactured(3.51), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{3}, id = example1_manufactured(4.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13)
    );

julia> cases1b = (
    (fe = Lagrange{1}, id = example1_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example1_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example1_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example1_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example1_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example1_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results1a = run_cases(Scheme2(), cases1a)
6-element Vector{ConvergenceResults}:
 ConvergenceResults: t_end=1.0, example1_manufactured(1.76), Lagrange{1}, Scheme2(), sum(walltime)=123.27 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.16e-02  0.00    3.11e-02  0.00    6.68e-02  0.00    4.64e-02  0.00    4.27e-02  0.00    1.13e-01  0.00    1.4454e+00
   8   -2.50  -13.00    3.01e-03  1.95    7.97e-03  1.96    1.51e-02  2.14    1.07e-02  2.12    1.10e-02  1.96    2.58e-02  2.13    5.7818e+00
  16   -3.50  -13.00    7.82e-04  1.94    2.06e-03  1.95    3.56e-03  2.09    2.56e-03  2.06    2.84e-03  1.95    6.12e-03  2.08    2.2854e+01
  32   -4.50  -13.00    2.12e-04  1.88    5.44e-04  1.92    8.57e-04  2.05    6.28e-04  2.03    7.56e-04  1.91    1.48e-03  2.04    9.3185e+01

 ConvergenceResults: t_end=1.0, example1_manufactured(2.4), Lagrange{1}, Scheme2(), sum(walltime)=125.24 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.83e-02  0.00    4.82e-02  0.00    6.67e-02  0.00    4.95e-02  0.00    6.65e-02  0.00    1.16e-01  0.00    1.4904e+00
   8   -2.50  -13.00    4.58e-03  2.00    1.20e-02  2.01    1.50e-02  2.15    1.14e-02  2.11    1.66e-02  2.01    2.64e-02  2.14    5.8608e+00
  16   -3.50  -13.00    1.15e-03  2.00    2.99e-03  2.00    3.60e-03  2.06    2.79e-03  2.04    4.13e-03  2.00    6.39e-03  2.05    2.3356e+01
  32   -4.50  -13.00    2.87e-04  2.00    7.46e-04  2.00    8.87e-04  2.02    6.92e-04  2.01    1.03e-03  2.00    1.58e-03  2.02    9.4528e+01

 ConvergenceResults: t_end=1.0, example1_manufactured(2.58), Lagrange{2}, Scheme2(), sum(walltime)=235.33 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    5.82e-04  0.00    1.45e-03  0.00    4.14e-03  0.00    3.75e-03  0.00    2.03e-03  0.00    7.89e-03  0.00    2.3307e+00
   8   -2.50  -13.00    7.71e-05  2.92    1.93e-04  2.91    6.15e-04  2.75    5.23e-04  2.84    2.70e-04  2.91    1.14e-03  2.79    9.3783e+00
  16   -3.50  -13.00    1.02e-05  2.92    2.53e-05  2.93    8.37e-05  2.88    6.84e-05  2.93    3.55e-05  2.93    1.52e-04  2.90    3.9344e+01
  32   -4.50  -13.00    1.36e-06  2.91    3.30e-06  2.94    1.10e-05  2.93    8.77e-06  2.96    4.66e-06  2.93    1.98e-05  2.94    1.8428e+02

 ConvergenceResults: t_end=1.0, example1_manufactured(3.4), Lagrange{2}, Scheme2(), sum(walltime)=237.95 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.34e-03  0.00    3.34e-03  0.00    5.06e-03  0.00    3.70e-03  0.00    4.68e-03  0.00    8.75e-03  0.00    2.3455e+00
   8   -2.50  -13.00    1.67e-04  3.00    4.18e-04  3.00    6.40e-04  2.98    4.84e-04  2.93    5.85e-04  3.00    1.12e-03  2.96    9.6234e+00
  16   -3.50  -13.00    2.09e-05  3.00    5.23e-05  3.00    8.32e-05  2.94    6.16e-05  2.98    7.32e-05  3.00    1.45e-04  2.96    4.0869e+01
  32   -4.50  -13.00    2.61e-06  3.00    6.53e-06  3.00    1.06e-05  2.97    7.74e-06  2.99    9.15e-06  3.00    1.84e-05  2.98    1.8512e+02

 ConvergenceResults: t_end=1.0, example1_manufactured(3.51), Lagrange{3}, Scheme2(), sum(walltime)=477.54 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    2.85e-05  0.00    7.18e-05  0.00    4.05e-04  0.00    1.37e-04  0.00    1.00e-04  0.00    5.42e-04  0.00    3.6533e+00
   8   -2.50  -13.00    1.95e-06  3.87    4.90e-06  3.87    4.05e-05  3.32    1.13e-05  3.60    6.85e-06  3.87    5.17e-05  3.39    1.5538e+01
  16   -3.50  -13.00    1.31e-07  3.89    3.29e-07  3.90    2.54e-06  3.99    7.14e-07  3.98    4.60e-07  3.89    3.25e-06  3.99    6.6985e+01
  32   -4.50  -13.00    8.93e-09  3.88    2.19e-08  3.91    1.49e-07  4.09    4.31e-08  4.05    3.08e-08  3.90    1.92e-07  4.08    3.9137e+02

 ConvergenceResults: t_end=1.0, example1_manufactured(4.4), Lagrange{3}, Scheme2(), sum(walltime)=479.08 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    8.89e-05  0.00    2.24e-04  0.00    5.89e-04  0.00    1.47e-04  0.00    3.13e-04  0.00    7.35e-04  0.00    3.6705e+00
   8   -2.50  -13.00    5.58e-06  4.00    1.40e-05  4.00    1.01e-04  2.55    2.41e-05  2.61    1.96e-05  4.00    1.25e-04  2.56    1.6176e+01
  16   -3.50  -13.00    3.50e-07  3.99    8.77e-07  4.00    7.05e-06  3.84    1.69e-06  3.84    1.23e-06  4.00    8.74e-06  3.84    6.7881e+01
  32   -4.50  -13.00    2.20e-08  3.99    5.48e-08  4.00    4.16e-07  4.08    1.00e-07  4.08    7.69e-08  4.00    5.16e-07  4.08    3.9136e+02

julia> results1b = run_cases(Scheme2(), cases1b)

julia> cases2a = (
    (fe = Lagrange{1}, id = example2_manufactured(1.76), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{1}, id = example2_manufactured(2.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{2}, id = example2_manufactured(2.58), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{2}, id = example2_manufactured(3.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{3}, id = example2_manufactured(3.51), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{3}, id = example2_manufactured(4.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13)
    );

julia> cases2b = (
    (fe = Lagrange{1}, id = example2_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example2_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example2_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example2_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example2_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example2_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results2a = run_cases(Scheme2(), cases2a)
6-element Vector{ConvergenceResults}:
 ConvergenceResults: t_end=1.0, example2_manufactured(1.76), Lagrange{1}, Scheme2(), sum(walltime)=116.50 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.09e-02  0.00    3.08e-02  0.00    6.68e-02  0.00    4.64e-02  0.00    4.17e-02  0.00    1.13e-01  0.00    1.3682e+00
   8   -2.50  -13.00    2.80e-03  1.96    7.90e-03  1.96    1.51e-02  2.14    1.07e-02  2.12    1.07e-02  1.96    2.58e-02  2.13    5.4755e+00
  16   -3.50  -13.00    7.23e-04  1.95    2.04e-03  1.95    3.56e-03  2.09    2.56e-03  2.06    2.76e-03  1.95    6.12e-03  2.08    2.1689e+01
  32   -4.50  -13.00    1.94e-04  1.90    5.38e-04  1.92    8.56e-04  2.06    6.28e-04  2.03    7.32e-04  1.92    1.48e-03  2.04    8.7972e+01

 ConvergenceResults: t_end=1.0, example2_manufactured(2.4), Lagrange{1}, Scheme2(), sum(walltime)=116.51 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.81e-02  0.00    4.56e-02  0.00    6.67e-02  0.00    4.95e-02  0.00    6.13e-02  0.00    1.16e-01  0.00    1.3734e+00
   8   -2.50  -13.00    4.69e-03  1.95    1.12e-02  2.02    1.50e-02  2.15    1.14e-02  2.11    1.54e-02  1.99    2.65e-02  2.14    5.4754e+00
  16   -3.50  -13.00    1.19e-03  1.98    2.80e-03  2.01    3.61e-03  2.06    2.79e-03  2.04    3.86e-03  2.00    6.40e-03  2.05    2.1718e+01
  32   -4.50  -13.00    2.98e-04  1.99    7.00e-04  2.00    8.88e-04  2.02    6.92e-04  2.01    9.64e-04  2.00    1.58e-03  2.02    8.7948e+01

 ConvergenceResults: t_end=1.0, example2_manufactured(2.58), Lagrange{2}, Scheme2(), sum(walltime)=221.37 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    5.80e-04  0.00    1.45e-03  0.00    4.14e-03  0.00    3.75e-03  0.00    2.03e-03  0.00    7.89e-03  0.00    2.1390e+00
   8   -2.50  -13.00    7.71e-05  2.91    1.93e-04  2.91    6.15e-04  2.75    5.23e-04  2.84    2.70e-04  2.91    1.14e-03  2.79    8.5903e+00
  16   -3.50  -13.00    1.02e-05  2.92    2.53e-05  2.93    8.37e-05  2.88    6.84e-05  2.93    3.55e-05  2.93    1.52e-04  2.90    3.6774e+01
  32   -4.50  -13.00    1.35e-06  2.91    3.30e-06  2.94    1.10e-05  2.93    8.77e-06  2.96    4.65e-06  2.93    1.98e-05  2.94    1.7386e+02

 ConvergenceResults: t_end=1.0, example2_manufactured(3.4), Lagrange{2}, Scheme2(), sum(walltime)=222.10 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.35e-03  0.00    3.34e-03  0.00    5.06e-03  0.00    3.70e-03  0.00    4.69e-03  0.00    8.75e-03  0.00    2.1443e+00
   8   -2.50  -13.00    1.67e-04  3.01    4.18e-04  3.00    6.40e-04  2.98    4.84e-04  2.93    5.85e-04  3.00    1.12e-03  2.96    8.8354e+00
  16   -3.50  -13.00    2.09e-05  3.00    5.23e-05  3.00    8.32e-05  2.94    6.16e-05  2.98    7.32e-05  3.00    1.45e-04  2.96    3.7739e+01
  32   -4.50  -13.00    2.61e-06  3.00    6.53e-06  3.00    1.06e-05  2.97    7.74e-06  2.99    9.15e-06  3.00    1.84e-05  2.98    1.7339e+02

 ConvergenceResults: t_end=1.0, example2_manufactured(3.51), Lagrange{3}, Scheme2(), sum(walltime)=457.59 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    2.95e-05  0.00    7.15e-05  0.00    4.05e-04  0.00    1.37e-04  0.00    1.00e-04  0.00    5.42e-04  0.00    3.4082e+00
   8   -2.50  -13.00    2.00e-06  3.88    4.90e-06  3.87    4.05e-05  3.32    1.13e-05  3.60    6.84e-06  3.87    5.17e-05  3.39    1.4516e+01
  16   -3.50  -13.00    1.31e-07  3.93    3.29e-07  3.89    2.54e-06  3.99    7.14e-07  3.98    4.60e-07  3.89    3.25e-06  3.99    6.3055e+01
  32   -4.50  -13.00    8.86e-09  3.89    2.19e-08  3.91    1.49e-07  4.09    4.31e-08  4.05    3.08e-08  3.90    1.92e-07  4.08    3.7661e+02

 ConvergenceResults: t_end=1.0, example2_manufactured(4.4), Lagrange{3}, Scheme2(), sum(walltime)=459.39 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.02e-04  0.00    2.21e-04  0.00    5.88e-04  0.00    1.47e-04  0.00    3.19e-04  0.00    7.35e-04  0.00    3.4960e+00
   8   -2.50  -13.00    6.28e-06  4.02    1.40e-05  3.98    1.01e-04  2.55    2.41e-05  2.61    2.01e-05  3.99    1.25e-04  2.56    1.4941e+01
  16   -3.50  -13.00    3.65e-07  4.11    8.76e-07  4.00    7.05e-06  3.84    1.69e-06  3.84    1.23e-06  4.03    8.74e-06  3.84    6.3833e+01
  32   -4.50  -13.00    2.25e-08  4.02    5.48e-08  4.00    4.16e-07  4.08    1.00e-07  4.08    7.70e-08  4.00    5.16e-07  4.08    3.7712e+02

julia> results2b = run_cases(Scheme2(), cases2b)

julia> cases3a = (
    (fe = Lagrange{1}, id = example3_manufactured(1.76), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{1}, id = example3_manufactured(2.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{2}, id = example3_manufactured(2.58), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{2}, id = example3_manufactured(3.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{3}, id = example3_manufactured(3.51), Nx = [2^i for i in 2:5], τ = 2.0^-13),
    (fe = Lagrange{3}, id = example3_manufactured(4.4 ), Nx = [2^i for i in 2:5], τ = 2.0^-13)
    );

julia> cases3b = (
    (fe = Lagrange{1}, id = example3_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example3_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example3_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example3_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example3_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example3_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results3a = run_cases(Scheme2(), cases3a)
6-element Vector{ConvergenceResults}:
 ConvergenceResults: t_end=1.0, example3_manufactured(1.76), Lagrange{1}, Scheme2(), sum(walltime)=114.57 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.19e-02  0.00    3.17e-02  0.00    2.21e-02  0.00    2.60e-02  0.00    4.36e-02  0.00    4.81e-02  0.00    1.3289e+00
   8   -2.50  -13.00    3.04e-03  1.97    8.19e-03  1.95    5.08e-03  2.12    6.07e-03  2.10    1.12e-02  1.96    1.11e-02  2.11    5.2355e+00
  16   -3.50  -13.00    7.90e-04  1.94    2.15e-03  1.93    1.22e-03  2.05    1.49e-03  2.02    2.94e-03  1.93    2.71e-03  2.04    2.1059e+01
  32   -4.50  -13.00    2.45e-04  1.69    5.91e-04  1.87    3.22e-04  1.93    3.82e-04  1.96    8.19e-04  1.84    7.04e-04  1.95    8.6944e+01

 ConvergenceResults: t_end=1.0, example3_manufactured(2.4), Lagrange{1}, Scheme2(), sum(walltime)=114.62 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.87e-02  0.00    4.88e-02  0.00    2.20e-02  0.00    2.79e-02  0.00    6.75e-02  0.00    4.99e-02  0.00    1.3384e+00
   8   -2.50  -13.00    4.65e-03  2.01    1.21e-02  2.01    5.05e-03  2.12    6.54e-03  2.09    1.68e-02  2.01    1.16e-02  2.11    5.2396e+00
  16   -3.50  -13.00    1.16e-03  2.00    3.03e-03  2.00    1.22e-03  2.05    1.61e-03  2.03    4.19e-03  2.00    2.83e-03  2.03    2.1075e+01
  32   -4.50  -13.00    2.90e-04  2.00    7.56e-04  2.00    3.02e-04  2.02    4.00e-04  2.01    1.05e-03  2.00    7.01e-04  2.01    8.6965e+01

 ConvergenceResults: t_end=1.0, example3_manufactured(2.58), Lagrange{2}, Scheme2(), sum(walltime)=217.58 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    5.82e-04  0.00    1.45e-03  0.00    1.41e-03  0.00    2.44e-03  0.00    2.03e-03  0.00    3.85e-03  0.00    2.0390e+00
   8   -2.50  -13.00    7.72e-05  2.91    1.93e-04  2.91    2.03e-04  2.79    3.31e-04  2.88    2.70e-04  2.91    5.34e-04  2.85    8.2508e+00
  16   -3.50  -13.00    1.03e-05  2.91    2.53e-05  2.93    2.73e-05  2.89    4.26e-05  2.96    3.56e-05  2.92    6.99e-05  2.93    3.5815e+01
  32   -4.50  -13.00    1.44e-06  2.84    3.32e-06  2.93    3.60e-06  2.92    5.41e-06  2.98    4.76e-06  2.90    9.01e-06  2.96    1.7148e+02

 ConvergenceResults: t_end=1.0, example3_manufactured(3.4), Lagrange{2}, Scheme2(), sum(walltime)=217.53 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    1.34e-03  0.00    3.34e-03  0.00    1.66e-03  0.00    2.35e-03  0.00    4.68e-03  0.00    4.02e-03  0.00    2.0357e+00
   8   -2.50  -13.00    1.67e-04  3.00    4.18e-04  3.00    2.10e-04  2.98    3.12e-04  2.92    5.85e-04  3.00    5.22e-04  2.94    8.2421e+00
  16   -3.50  -13.00    2.09e-05  3.00    5.23e-05  3.00    2.64e-05  2.99    3.96e-05  2.98    7.32e-05  3.00    6.60e-05  2.98    3.5889e+01
  32   -4.50  -13.00    2.61e-06  3.00    6.53e-06  3.00    3.31e-06  3.00    4.96e-06  2.99    9.15e-06  3.00    8.27e-06  3.00    1.7137e+02

 ConvergenceResults: t_end=1.0, example3_manufactured(3.51), Lagrange{3}, Scheme2(), sum(walltime)=445.88 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    2.86e-05  0.00    7.19e-05  0.00    4.64e-05  0.00    6.94e-05  0.00    1.00e-04  0.00    1.16e-04  0.00    3.0673e+00
   8   -2.50  -13.00    1.96e-06  3.87    4.90e-06  3.87    3.12e-06  3.90    4.29e-06  4.02    6.86e-06  3.87    7.41e-06  3.97    1.3058e+01
  16   -3.50  -13.00    1.31e-07  3.89    3.29e-07  3.90    2.05e-07  3.93    2.69e-07  4.00    4.61e-07  3.90    4.73e-07  3.97    6.0918e+01
  32   -4.50  -13.00    9.06e-09  3.86    2.19e-08  3.91    1.34e-08  3.93    1.69e-08  3.99    3.10e-08  3.89    3.03e-08  3.97    3.6884e+02

 ConvergenceResults: t_end=1.0, example3_manufactured(4.4), Lagrange{3}, Scheme2(), sum(walltime)=446.00 s
  Nx   log₂h   log₂τ    L∞L²_V    rate    L∞L²_U    rate    L∞L²_R    rate    L∞L²_Z    rate    L∞L²_VU   rate    L∞L²_RZ   rate    walltime[s]
   4   -1.50  -13.00    8.93e-05  0.00    2.24e-04  0.00    5.01e-05  0.00    5.59e-05  0.00    3.13e-04  0.00    1.00e-04  0.00    3.0494e+00
   8   -2.50  -13.00    5.59e-06  4.00    1.40e-05  4.00    2.96e-06  4.08    3.38e-06  4.05    1.96e-05  4.00    6.04e-06  4.06    1.3059e+01
  16   -3.50  -13.00    3.51e-07  3.99    8.77e-07  4.00    1.78e-07  4.06    2.09e-07  4.01    1.23e-06  4.00    3.68e-07  4.04    6.0811e+01
  32   -4.50  -13.00    2.21e-08  3.99    5.48e-08  4.00    1.09e-08  4.03    1.30e-08  4.00    7.69e-08  4.00    2.27e-08  4.02    3.6908e+02

julia> results3b = run_cases(Scheme2(), cases3b)

julia> cases4 = (
    (fe = Lagrange{1}, id = example1_manufactured(2.4 ), Nx = 2^9, τ = [2.0^-i for i in 2:5]),
    (fe = Lagrange{1}, id = example2_manufactured(2.4 ), Nx = 2^9, τ = [2.0^-i for i in 2:5]),
    (fe = Lagrange{1}, id = example3_manufactured(2.4 ), Nx = 2^9, τ = [2.0^-i for i in 2:5])
    );

julia> results4 = run_cases(Scheme2(), cases4)

julia> versioninfo()
Julia Version 1.12.6
Commit 15346901f00 (2026-04-09 19:20 UTC)
Build Info:
  Official https://julialang.org release
Platform Info:
  OS: Linux (x86_64-linux-gnu)
  CPU: 14 × Intel(R) Core(TM) Ultra 5 225H
  WORD_SIZE: 64
  LLVM: libLLVM-18.1.7 (ORCJIT, arrowlake)
  GC: Built with stock GC
Threads: 1 default, 1 interactive, 1 GC (on 14 virtual cores)
```