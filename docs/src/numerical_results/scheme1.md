# Scheme 1
```julia-repl
julia> using WaveAcoustics

julia> cases1 = (
    (fe = Lagrange{1}, id = example1_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example1_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example1_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example1_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example1_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example1_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results1 = run_cases(Scheme1(), cases1)
6-element Vector{ConvergenceResults}:
 ConvergenceResults: t_end=1.0, example1_manufactured(1.76), Lagrange{1}, Scheme1(), elapsed=1127.42 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.16e-02  0.000    3.11e-02  0.000    6.68e-02  0.000    4.64e-02  0.000
   8   -2.50  -15.00    3.01e-03  1.953    7.97e-03  1.962    1.51e-02  2.142    1.07e-02  2.122
  16   -3.50  -15.00    7.82e-04  1.943    2.06e-03  1.952    3.56e-03  2.089    2.56e-03  2.057
  32   -4.50  -15.00    2.12e-04  1.881    5.44e-04  1.922    8.57e-04  2.055    6.28e-04  2.028
  64   -5.50  -15.00    7.13e-05  1.574    1.52e-04  1.841    2.12e-04  2.011    1.58e-04  1.996

 ConvergenceResults: t_end=1.0, example1_manufactured(2.4), Lagrange{1}, Scheme1(), elapsed=1141.25 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.83e-02  0.000    4.82e-02  0.000    6.67e-02  0.000    4.95e-02  0.000
   8   -2.50  -15.00    4.58e-03  2.000    1.20e-02  2.010    1.50e-02  2.153    1.14e-02  2.113
  16   -3.50  -15.00    1.15e-03  1.997    2.99e-03  2.003    3.60e-03  2.057    2.79e-03  2.036
  32   -4.50  -15.00    2.87e-04  1.999    7.46e-04  2.001    8.87e-04  2.022    6.92e-04  2.012
  64   -5.50  -15.00    7.17e-05  2.000    1.86e-04  2.001    2.20e-04  2.009    1.72e-04  2.004

 ConvergenceResults: t_end=1.0, example1_manufactured(2.58), Lagrange{2}, Scheme1(), elapsed=2868.79 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    5.82e-04  0.000    1.45e-03  0.000    4.14e-03  0.000    3.75e-03  0.000
   8   -2.50  -15.00    7.71e-05  2.916    1.93e-04  2.911    6.15e-04  2.751    5.23e-04  2.843
  16   -3.50  -15.00    1.02e-05  2.920    2.53e-05  2.929    8.37e-05  2.877    6.84e-05  2.934
  32   -4.50  -15.00    1.36e-06  2.906    3.30e-06  2.939    1.10e-05  2.928    8.77e-06  2.964
  64   -5.50  -15.00    1.98e-07  2.780    4.34e-07  2.928    1.43e-06  2.948    1.12e-06  2.974

 ConvergenceResults: t_end=1.0, example1_manufactured(3.4), Lagrange{2}, Scheme1(), elapsed=2870.39 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.34e-03  0.000    3.34e-03  0.000    5.06e-03  0.000    3.70e-03  0.000
   8   -2.50  -15.00    1.67e-04  3.000    4.18e-04  2.999    6.40e-04  2.982    4.84e-04  2.931
  16   -3.50  -15.00    2.09e-05  3.000    5.23e-05  3.000    8.32e-05  2.944    6.16e-05  2.976
  32   -4.50  -15.00    2.61e-06  3.000    6.53e-06  3.000    1.06e-05  2.969    7.74e-06  2.992
  64   -5.50  -15.00    3.27e-07  3.000    8.17e-07  3.000    1.34e-06  2.989    9.68e-07  2.998

 ConvergenceResults: t_end=1.0, example1_manufactured(3.51), Lagrange{3}, Scheme1(), elapsed=8824.31 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    2.85e-05  0.000    7.18e-05  0.000    4.05e-04  0.000    1.37e-04  0.000
   8   -2.50  -15.00    1.95e-06  3.868    4.90e-06  3.873    4.05e-05  3.325    1.13e-05  3.598
  16   -3.50  -15.00    1.31e-07  3.893    3.29e-07  3.895    2.54e-06  3.994    7.14e-07  3.984
  32   -4.50  -15.00    8.78e-09  3.902    2.19e-08  3.910    1.49e-07  4.094    4.30e-08  4.052
  64   -5.50  -15.00    5.99e-10  3.874    1.45e-09  3.919    8.81e-09  4.078    2.62e-09  4.039

 ConvergenceResults: t_end=1.0, example1_manufactured(4.4), Lagrange{3}, Scheme1(), elapsed=8883.51 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    8.89e-05  0.000    2.24e-04  0.000    5.89e-04  0.000    1.47e-04  0.000
   8   -2.50  -15.00    5.58e-06  3.995    1.40e-05  3.996    1.01e-04  2.545    2.41e-05  2.608
  16   -3.50  -15.00    3.50e-07  3.992    8.77e-07  3.999    7.05e-06  3.838    1.69e-06  3.835
  32   -4.50  -15.00    2.19e-08  3.998    5.48e-08  4.000    4.16e-07  4.083    1.00e-07  4.078
  64   -5.50  -15.00    1.38e-09  3.991    3.43e-09  4.000    2.43e-08  4.099    5.84e-09  4.097


julia> cases2 = (
    (fe = Lagrange{1}, id = example2_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example2_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example2_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example2_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example2_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example2_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results2 = run_cases(Scheme1(), cases2)
6-element Vector{ConvergenceResults}:
 ConvergenceResults: t_end=1.0, example2_manufactured(1.76), Lagrange{1}, Scheme1(), elapsed=1054.06 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.09e-02  0.000    3.08e-02  0.000    6.68e-02  0.000    4.64e-02  0.000
   8   -2.50  -15.00    2.80e-03  1.960    7.90e-03  1.964    1.51e-02  2.142    1.07e-02  2.122
  16   -3.50  -15.00    7.23e-04  1.954    2.04e-03  1.953    3.56e-03  2.089    2.56e-03  2.057
  32   -4.50  -15.00    1.94e-04  1.898    5.38e-04  1.924    8.56e-04  2.055    6.28e-04  2.028
  64   -5.50  -15.00    6.70e-05  1.533    1.50e-04  1.846    2.12e-04  2.013    1.57e-04  1.996

 ConvergenceResults: t_end=1.0, example2_manufactured(2.4), Lagrange{1}, Scheme1(), elapsed=1054.23 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.81e-02  0.000    4.56e-02  0.000    6.67e-02  0.000    4.95e-02  0.000
   8   -2.50  -15.00    4.69e-03  1.949    1.12e-02  2.019    1.50e-02  2.153    1.14e-02  2.113
  16   -3.50  -15.00    1.19e-03  1.983    2.80e-03  2.005    3.61e-03  2.057    2.79e-03  2.036
  32   -4.50  -15.00    2.98e-04  1.995    7.00e-04  2.002    8.88e-04  2.022    6.92e-04  2.012
  64   -5.50  -15.00    7.45e-05  1.998    1.75e-04  2.001    2.21e-04  2.009    1.72e-04  2.004

 ConvergenceResults: t_end=1.0, example2_manufactured(2.58), Lagrange{2}, Scheme1(), elapsed=2691.65 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    5.80e-04  0.000    1.45e-03  0.000    4.14e-03  0.000    3.75e-03  0.000
   8   -2.50  -15.00    7.71e-05  2.912    1.93e-04  2.911    6.15e-04  2.751    5.23e-04  2.843
  16   -3.50  -15.00    1.02e-05  2.921    2.53e-05  2.929    8.37e-05  2.877    6.84e-05  2.934
  32   -4.50  -15.00    1.35e-06  2.912    3.30e-06  2.940    1.10e-05  2.928    8.77e-06  2.964
  64   -5.50  -15.00    1.93e-07  2.810    4.31e-07  2.936    1.42e-06  2.948    1.12e-06  2.974

 ConvergenceResults: t_end=1.0, example2_manufactured(3.4), Lagrange{2}, Scheme1(), elapsed=2701.01 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.35e-03  0.000    3.34e-03  0.000    5.06e-03  0.000    3.70e-03  0.000
   8   -2.50  -15.00    1.67e-04  3.012    4.18e-04  2.998    6.40e-04  2.982    4.84e-04  2.932
  16   -3.50  -15.00    2.09e-05  3.000    5.23e-05  3.000    8.32e-05  2.944    6.16e-05  2.976
  32   -4.50  -15.00    2.61e-06  3.000    6.53e-06  3.000    1.06e-05  2.969    7.74e-06  2.992
  64   -5.50  -15.00    3.27e-07  3.000    8.17e-07  3.000    1.34e-06  2.989    9.68e-07  2.998

 ConvergenceResults: t_end=1.0, example2_manufactured(3.51), Lagrange{3}, Scheme1(), elapsed=8685.07 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    2.95e-05  0.000    7.15e-05  0.000    4.05e-04  0.000    1.37e-04  0.000
   8   -2.50  -15.00    2.00e-06  3.881    4.90e-06  3.869    4.05e-05  3.325    1.13e-05  3.598
  16   -3.50  -15.00    1.31e-07  3.932    3.29e-07  3.895    2.54e-06  3.994    7.14e-07  3.984
  32   -4.50  -15.00    8.78e-09  3.902    2.19e-08  3.910    1.49e-07  4.094    4.30e-08  4.052
  64   -5.50  -15.00    5.92e-10  3.889    1.45e-09  3.919    8.81e-09  4.078    2.62e-09  4.039

 ConvergenceResults: t_end=1.0, example2_manufactured(4.4), Lagrange{3}, Scheme1(), elapsed=8465.05 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.02e-04  0.000    2.21e-04  0.000    5.88e-04  0.000    1.47e-04  0.000
   8   -2.50  -15.00    6.28e-06  4.022    1.40e-05  3.985    1.01e-04  2.545    2.41e-05  2.608
  16   -3.50  -15.00    3.65e-07  4.106    8.76e-07  3.997    7.05e-06  3.838    1.69e-06  3.835
  32   -4.50  -15.00    2.24e-08  4.029    5.48e-08  3.999    4.16e-07  4.083    1.00e-07  4.078
  64   -5.50  -15.00    1.41e-09  3.982    3.43e-09  3.999    2.43e-08  4.099    5.84e-09  4.097


julia> cases3 = (
    (fe = Lagrange{1}, id = example3_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example3_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example3_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example3_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example3_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example3_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results3 = run_cases(Scheme1(), cases3)
6-element Vector{ConvergenceResults}:
 ConvergenceResults: t_end=1.0, example3_manufactured(1.76), Lagrange{1}, Scheme1(), elapsed=1036.23 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.19e-02  0.000    3.17e-02  0.000    2.21e-02  0.000    2.60e-02  0.000
   8   -2.50  -15.00    3.04e-03  1.969    8.19e-03  1.951    5.08e-03  2.124    6.07e-03  2.100
  16   -3.50  -15.00    7.90e-04  1.942    2.15e-03  1.928    1.22e-03  2.053    1.49e-03  2.025
  32   -4.50  -15.00    2.45e-04  1.690    5.91e-04  1.865    3.22e-04  1.928    3.82e-04  1.963
  64   -5.50  -15.00    1.29e-04  0.929    1.80e-04  1.717    1.07e-04  1.589    1.12e-04  1.776

 ConvergenceResults: t_end=1.0, example3_manufactured(2.4), Lagrange{1}, Scheme1(), elapsed=1037.85 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.87e-02  0.000    4.88e-02  0.000    2.20e-02  0.000    2.79e-02  0.000
   8   -2.50  -15.00    4.65e-03  2.007    1.21e-02  2.009    5.05e-03  2.124    6.54e-03  2.093
  16   -3.50  -15.00    1.16e-03  2.000    3.03e-03  2.003    1.22e-03  2.047    1.61e-03  2.026
  32   -4.50  -15.00    2.90e-04  1.999    7.56e-04  2.002    3.02e-04  2.018    4.00e-04  2.007
  64   -5.50  -15.00    7.27e-05  1.999    1.89e-04  2.002    7.51e-05  2.008    9.97e-05  2.002

 ConvergenceResults: t_end=1.0, example3_manufactured(2.58), Lagrange{2}, Scheme1(), elapsed=2653.57 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    5.82e-04  0.000    1.45e-03  0.000    1.41e-03  0.000    2.44e-03  0.000
   8   -2.50  -15.00    7.72e-05  2.913    1.93e-04  2.911    2.03e-04  2.795    3.31e-04  2.881
  16   -3.50  -15.00    1.03e-05  2.910    2.53e-05  2.928    2.73e-05  2.893    4.26e-05  2.958
  32   -4.50  -15.00    1.44e-06  2.840    3.32e-06  2.930    3.60e-06  2.922    5.41e-06  2.978
  64   -5.50  -15.00    2.95e-07  2.284    4.54e-07  2.873    4.94e-07  2.868    6.95e-07  2.961

 ConvergenceResults: t_end=1.0, example3_manufactured(3.4), Lagrange{2}, Scheme1(), elapsed=2652.25 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    1.34e-03  0.000    3.34e-03  0.000    1.66e-03  0.000    2.35e-03  0.000
   8   -2.50  -15.00    1.67e-04  2.999    4.18e-04  2.999    2.10e-04  2.983    3.12e-04  2.917
  16   -3.50  -15.00    2.09e-05  3.000    5.23e-05  3.000    2.64e-05  2.994    3.96e-05  2.978
  32   -4.50  -15.00    2.61e-06  3.000    6.53e-06  3.000    3.31e-06  2.998    4.96e-06  2.994
  64   -5.50  -15.00    3.27e-07  3.000    8.17e-07  3.000    4.13e-07  2.999    6.21e-07  2.999

 ConvergenceResults: t_end=1.0, example3_manufactured(3.51), Lagrange{3}, Scheme1(), elapsed=8666.03 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    2.86e-05  0.000    7.19e-05  0.000    4.64e-05  0.000    6.94e-05  0.000
   8   -2.50  -15.00    1.96e-06  3.871    4.90e-06  3.874    3.12e-06  3.896    4.29e-06  4.016
  16   -3.50  -15.00    1.31e-07  3.895    3.29e-07  3.896    2.05e-07  3.930    2.69e-07  3.997
  32   -4.50  -15.00    8.81e-09  3.899    2.19e-08  3.910    1.34e-08  3.933    1.69e-08  3.991
  64   -5.50  -15.00    6.19e-10  3.831    1.45e-09  3.913    8.79e-10  3.930    1.07e-09  3.985

 ConvergenceResults: t_end=1.0, example3_manufactured(4.4), Lagrange{3}, Scheme1(), elapsed=8046.98 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
   4   -1.50  -15.00    8.93e-05  0.000    2.24e-04  0.000    5.01e-05  0.000    5.59e-05  0.000
   8   -2.50  -15.00    5.59e-06  3.998    1.40e-05  3.996    2.96e-06  4.079    3.38e-06  4.050
  16   -3.50  -15.00    3.51e-07  3.995    8.77e-07  3.999    1.78e-07  4.057    2.09e-07  4.013
  32   -4.50  -15.00    2.19e-08  3.999    5.48e-08  4.000    1.09e-08  4.031    1.30e-08  4.003
  64   -5.50  -15.00    1.38e-09  3.990    3.43e-09  4.000    6.74e-10  4.014    8.15e-10  4.001


julia> cases4 = (
    (fe = Lagrange{1}, id = example1_manufactured(2.4 ), Nx = 2^9, τ = [2.0^-i for i in 2:5]),
    (fe = Lagrange{1}, id = example2_manufactured(2.4 ), Nx = 2^9, τ = [2.0^-i for i in 2:5]),
    (fe = Lagrange{1}, id = example3_manufactured(2.4 ), Nx = 2^9, τ = [2.0^-i for i in 2:5])
    );

julia> results4 = run_cases(Scheme1(), cases4)
3-element Vector{ConvergenceResults}:
 ConvergenceResults: t_end=1.0, example1_manufactured(2.4), Lagrange{1}, Scheme1(), elapsed=407.93 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
 512   -8.50   -2.00    1.01e-02  0.000    3.49e-03  0.000    7.12e-03  0.000    5.46e-03  0.000
 512   -8.50   -3.00    2.55e-03  1.982    8.86e-04  1.978    1.79e-03  1.992    1.37e-03  1.993
 512   -8.50   -4.00    6.37e-04  1.999    2.24e-04  1.981    4.49e-04  1.994    3.44e-04  1.998
 512   -8.50   -5.00    1.59e-04  2.001    5.74e-05  1.968    1.13e-04  1.996    8.60e-05  1.998

 ConvergenceResults: t_end=1.0, example2_manufactured(2.4), Lagrange{1}, Scheme1(), elapsed=392.39 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
 512   -8.50   -2.00    1.09e-02  0.000    3.51e-03  0.000    7.11e-03  0.000    5.41e-03  0.000
 512   -8.50   -3.00    2.76e-03  1.989    9.08e-04  1.949    1.79e-03  1.989    1.36e-03  1.991
 512   -8.50   -4.00    6.87e-04  2.005    2.31e-04  1.977    4.49e-04  1.994    3.41e-04  1.997
 512   -8.50   -5.00    1.71e-04  2.006    5.91e-05  1.966    1.13e-04  1.995    8.53e-05  1.997

 ConvergenceResults: t_end=1.0, example3_manufactured(2.4), Lagrange{1}, Scheme1(), elapsed=220.94 s
  Nx   log₂h   log₂τ    L∞L²_V    rate     L∞L²_U    rate     L∞L²_R    rate     L∞L²_Z    rate
 512   -8.50   -2.00    1.02e-02  0.000    3.92e-03  0.000    1.25e-03  0.000    7.27e-04  0.000
 512   -8.50   -3.00    2.59e-03  1.980    9.71e-04  2.013    3.65e-04  1.777    1.75e-04  2.057
 512   -8.50   -4.00    6.47e-04  1.998    2.41e-04  2.010    9.72e-05  1.911    4.33e-05  2.011
 512   -8.50   -5.00    1.62e-04  1.995    6.19e-05  1.961    2.49e-05  1.962    1.09e-05  1.985


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