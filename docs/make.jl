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
        assets = String[]
    ),
    pages = [
        "Home" => "index.md",
        "Theory" => "theory/model.md",
        "Methods" => [
            "Overview" => "methods/overview.md",
            "Method 1" => "methods/method1.md",
            "Method 2" => "methods/method2.md"
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
