using StreamingSampling
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(
    StreamingSampling,
    :DocTestSetup,
    :(using StreamingSampling);
    recursive = true,
)

ENV["BASE_PATH"] = joinpath(@__DIR__, "../") 

# Citations ####################################################################
bib = CitationBibliography(joinpath(@__DIR__, "citation.bib"))


# Generate examples ############################################################
const examples_path = joinpath(@__DIR__, "..", "examples")
const output_path   = joinpath(@__DIR__, "src/generated")
function create_examples(examples, examples_path, output_path)
    for (_, example_path) in examples
        s = split(example_path, "/")
        sub_path, file_name = string(s[1:end-1]...), s[end]
        example_filepath = joinpath(examples_path, example_path)
        Literate.markdown(example_filepath,
                          joinpath(output_path, sub_path),
                          documenter = true)
    end
    examples = [title => joinpath("generated", replace(example_path, ".jl" => ".md"))
                for (title, example_path) in examples]
    return examples
end

# Basic examples
examples = [
    "Simple example" => "simple-example/simple-example.jl",
]
basic_examples = create_examples(examples, examples_path, output_path)

# Make and deploy docs #########################################################

makedocs(
      root    =  joinpath(dirname(pathof(StreamingSampling)), "..", "docs"),
      source  = "src",
      build   = "build",
      clean   = true,
      doctest = true,
      modules = [StreamingSampling],
      repo    = "https://github.com/JuliaLabs/StreamingSampling.jl/blob/{commit}{path}#{line}",
      highlightsig = true,
      sitename = "StreamingSampling.jl",
      expandfirst = [],
      draft = false,
      pages = ["Home" => "index.md",
               "Install and run" => "install-and-run.md",
               "Basic examples" => basic_examples,
               "API" => "api.md"],
      format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://github.com/JuliaLabs/StreamingSampling.jl",
        assets = String[],
      ),
      plugins=[bib]
)

deploydocs(;
    repo = "https://github.com/JuliaLabs/StreamingSampling.jl",
    devbranch = "main",
    push_preview = true,
)

