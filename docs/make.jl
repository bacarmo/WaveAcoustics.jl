using WaveAcoustics
using Documenter

DocMeta.setdocmeta!(WaveAcoustics, :DocTestSetup, :(using WaveAcoustics); recursive=true)

makedocs(;
    modules=[WaveAcoustics],
    authors="Bruno Alves do Carmo <bruno.carmo@ppgi.ufrj.br>",
    sitename="WaveAcoustics.jl",
    format=Documenter.HTML(;
        canonical="https://bacarmo.github.io/WaveAcoustics.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bacarmo/WaveAcoustics.jl",
    devbranch="main",
)
