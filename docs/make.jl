using Documenter

push!(LOAD_PATH, "../src/")
using LazyAlgebra

DEPLOYDOCS = (get(ENV, "CI", nothing) == "true")

makedocs(
    sitename = "LazyAlgebra for Julia",
    format = Documenter.HTML(
        prettyurls = DEPLOYDOCS,
    ),
    authors = "Éric Thiébaut and contributors",
    pages = ["index.md", "install.md", "introduction.md",
             "vectors.md", "mappings.md"]
)

if DEPLOYDOCS
    deploydocs(
        repo = "github.com/emmt/LazyAlgebra.jl.git",
    )
end
