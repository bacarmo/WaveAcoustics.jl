# Scheme 3
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

julia> results1 = run_cases(Scheme3(), cases1)

julia> cases2 = (
    (fe = Lagrange{1}, id = example2_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example2_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example2_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example2_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example2_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example2_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results2 = run_cases(Scheme3(), cases2)

julia> cases3 = (
    (fe = Lagrange{1}, id = example3_manufactured(1.76), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{1}, id = example3_manufactured(2.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example3_manufactured(2.58), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{2}, id = example3_manufactured(3.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example3_manufactured(3.51), Nx = [2^i for i in 2:6], τ = 2.0^-15),
    (fe = Lagrange{3}, id = example3_manufactured(4.4 ), Nx = [2^i for i in 2:6], τ = 2.0^-15)
    );

julia> results3 = run_cases(Scheme3(), cases3)

julia> versioninfo()
```