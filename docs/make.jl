using WaveAcoustics
using Documenter

DocMeta.setdocmeta!(WaveAcoustics, :DocTestSetup, :(using WaveAcoustics); recursive = true)

makedocs(;
    modules = [WaveAcoustics],
    authors = "Bruno Alves do Carmo <bruno.carmo@ppgi.ufrj.br>",
    sitename = "WaveAcoustics.jl",
    format = Documenter.HTML(;
        canonical = "https://bacarmo.github.io/WaveAcoustics.jl",
        edit_link = "main",
        assets = String[],
        size_threshold = 300 * 1024,  # error threshold: 300 KiB
        size_threshold_warn = 200 * 1024,   # warning threshold: 200 KiB
        mathengine = MathJax3() # The options are either KaTeX (default), MathJax v2, or MathJax v3, enabled by passing an instance of KaTeX, MathJax2, or MathJax3 objects, respectively.
    ),
    pages = [
        "Home" => "index.md",
        "Model" => "model.md",
        "Approximation Problem" => [
            "schemes/scheme1.md",
            "schemes/scheme2.md",
            "schemes/scheme3.md"
        ],
        "Numerical Results" => [
            "numerical_results/scheme1.md",
            "numerical_results/scheme2.md",
            "numerical_results/scheme3.md"
        ],
        "API" => "api.md"
    ]
)

deploydocs(;
    repo = "github.com/bacarmo/WaveAcoustics.jl",
    devbranch = "main"
)
