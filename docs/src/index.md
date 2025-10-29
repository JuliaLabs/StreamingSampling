```@meta
CurrentModule = StreamingSampling
```

# StreamingSampling.jl

StreamingSampling is a Julia-based proof-of-concept implementation of a streamed variants of maximum-entropy sampling ([UPmaxentropy](https://www.rdocumentation.org/packages/sampling/versions/2.11/topics/UPmaxentropy)) and weighted sampling. It is designed to process large datasets stored on disk with minimal impact on RAM. The method begins by computing first-order inclusion probabilities using a [DPP](https://dahtah.github.io/Determinantal.jl/dev/)-based heuristic, and then feeds these probabilities into classical sampling algorithms to produce diverse samples.


