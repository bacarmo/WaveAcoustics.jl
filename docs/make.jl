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
        mathengine = MathJax3() # The options are either KaTeX (default), MathJax v2, or MathJax v3, enabled by passing an instance of KaTeX, MathJax2, or MathJax3 objects, respectively.
    ),
    pages = [
        "Home" => "index.md",
        "Model" => "theory/model.md",
        "Approximation Problem" => [
            "methods/method1.md",
            "methods/method2.md"
        ],
        "Examples" => [
            "Overview" => "examples/overview.md",
            "Example 1" => "examples/example1.md",
            "Example 2" => "examples/example2.md"
        ],
        "API Reference" => "api/api.md"
    ]
)

deploydocs(;
    repo = "github.com/bacarmo/WaveAcoustics.jl",
    devbranch = "main"
)
