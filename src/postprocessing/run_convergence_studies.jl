# ==============================================================================
# run_convergence_studies.jl
#
# Runs coupled and spatial convergence studies for the WaveAcoustics
# package and saves the results to a timestamped plain-text file in the package root.
#
# HOW TO RUN
# ----------
# 0. (First use only) Download the package and install all dependencies.
#
#    Option A — Using Git:
#
#       git clone https://github.com/bacarmo/WaveAcoustics.jl.git
#
#    Option B — Without Git:
#       Go to https://github.com/bacarmo/WaveAcoustics.jl,
#       click "Code" > "Download ZIP", and extract the folder.
#
#    Then, open a terminal at the package root and run:
#
#       julia
#       ] activate .
#       ] instantiate
#       <backspace>  (to exit Pkg mode)
#
#    This step is only required once per machine.
#
# 1. Open a terminal at the package root:
#
#       WaveAcoustics/   <-- open the terminal here
#       ├── Project.toml
#       └── src/
#           └── postprocessing/
#               └── run_convergence_studies.jl
#
# 2. Start Julia and activate the package environment:
#
#       julia
#       ] activate .
#       <backspace>  (to exit Pkg mode)
#
# 3. Load and run the script:
#
#       include("src/postprocessing/run_convergence_studies.jl")
#
#    To re-run after modifying CASES or the package source, just call:
#
#       include("src/postprocessing/run_convergence_studies.jl")
#
# 4. Results are saved to a timestamped .txt file in the package root, e.g.:
#
#       convergence_results_YYYY-MM-DD__HH-MM-SS.txt
# ==============================================================================

using Revise
using Dates
using Printf
using WaveAcoustics

# ── Configuration ──────────────────────────────────────────────────────────────

# Cases to run
#! format: off
const CASES = (
    (input_data = example1_manufactured(1.76), solver = CrankNicolson1(), fe = Lagrange{1}()),
    (input_data = example1_manufactured(2.4 ), solver = CrankNicolson1(), fe = Lagrange{1}()),
    (input_data = example1_manufactured(2.58), solver = CrankNicolson1(), fe = Lagrange{2}()),
    (input_data = example1_manufactured(3.4 ), solver = CrankNicolson1(), fe = Lagrange{2}()),
    (input_data = example1_manufactured(3.51), solver = CrankNicolson1(), fe = Lagrange{3}()),
    (input_data = example1_manufactured(4.4 ), solver = CrankNicolson1(), fe = Lagrange{3}()),
)

# Studies to run 
# Set `run = false` to skip a study.
# Adjust `cases` to run a subset; see CASES above for indices.
#
# Note: the temporal study fixes the spatial discretization to isolate convergence in time. 
# One case per solver suffices; Lagrange{1} is the cheapest.
const STUDIES = (
    coupled =  (run = true, cases = CASES, Nx_exp_range = 2:6),
    spatial =  (run = true, cases = CASES, Nx_exp_range = 2:6, τ_fixed = 2.0^(-15)),
    temporal = (run = true, cases = CASES[[1,2]], τ_exp_range = 2:5, Nx_fixed = 2^9),
)
# Output file - timestamp prevents overwriting previous runs
const OUTPUT_FILE = "convergence_results_$(Dates.format(now(), "yyyy-mm-dd__HH-MM-SS")).txt"

# ── Helpers ────────────────────────────────────────────────────────────────────

# Shared loop: warm-up, run, print, flush, error handling
function run_study!(io, label, cases, warmup_fn, study_fn)
    println("# $label convergence study")
    println("# Run started: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))\n")
    flush(io)

    for case in cases
        try
            warmup_fn(case)
            elapsed = @elapsed results = study_fn(case)
            print_convergence_table(results)
            @printf("  Elapsed: %.2f s\n", elapsed)
        catch e
            println("\n[ERROR] $(case.input_data.name) · $(typeof(case.solver)) · $(typeof(case.fe)):\n  $e\n")
        end
        flush(io)
    end

    println("\n# Run finished: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
end

# ── Study definitions ──────────────────────────────────────────────────────────

# Coupled study — refines Nx and τ simultaneously
function warmup_coupled(case)
    convergence_study_coupled(
        input_data   = case.input_data,
        solver       = case.solver,
        fe           = case.fe,
        Nx_exp_range = 2:2)
end

function study_coupled(case)
    convergence_study_coupled(
        input_data   = case.input_data,
        solver       = case.solver,
        fe           = case.fe,
        Nx_exp_range = STUDIES.coupled.Nx_exp_range)
end

# Spatial study — refines Nx with τ fixed
function warmup_spatial(case)
    convergence_study_spatial(
        input_data   = case.input_data,
        solver       = case.solver,
        fe           = case.fe,
        Nx_exp_range = 2:2,
        τ_fixed      = 2.0^(-3))
end

function study_spatial(case)
    convergence_study_spatial(
        input_data   = case.input_data,
        solver       = case.solver,
        fe           = case.fe,
        Nx_exp_range = STUDIES.spatial.Nx_exp_range,
        τ_fixed      = STUDIES.spatial.τ_fixed)
end

# Temporal study — refines τ with Nx fixed
function warmup_temporal(case)
    convergence_study_temporal(
        input_data   = case.input_data,
        solver       = case.solver,
        fe           = case.fe,
        τ_exp_range  = 2:2,
        Nx_fixed     = 4)
end

function study_temporal(case)
    convergence_study_temporal(
        input_data   = case.input_data,
        solver       = case.solver,
        fe           = case.fe,
        τ_exp_range  = STUDIES.temporal.τ_exp_range,
        Nx_fixed     = STUDIES.temporal.Nx_fixed)
end

# ── Main ───────────────────────────────────────────────────────────────────────

# open: creates the file and closes it on exit, even if an exception is raised
open(OUTPUT_FILE, "w") do io
    # redirect_stdout: redirects print/println/@printf to `io` instead of the terminal
    redirect_stdout(io) do
        STUDIES.coupled.run  && run_study!(io, "Coupled",  STUDIES.coupled.cases,  warmup_coupled,  study_coupled)
        STUDIES.spatial.run  && run_study!(io, "Spatial",  STUDIES.spatial.cases,  warmup_spatial,  study_spatial)
        STUDIES.temporal.run && run_study!(io, "Temporal", STUDIES.temporal.cases, warmup_temporal, study_temporal)
    end
end
#! format: on