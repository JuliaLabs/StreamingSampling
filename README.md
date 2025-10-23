# StreamingSampling

StreamingSampling is a Julia-based proof-of-concept implementation of a streamed variant of maximum-entropy sampling (UPmaxentropy). It is designed to process large datasets stored on disk with minimal impact on RAM. The method first computes first-order inclusion probabilities using a DPP-based heuristic, and then feeds these probabilities into the classical UPmaxentropy algorithm to produce diverse samples.

## References

1. UPmaxentropy — Original maximum-entropy sampling method ([Link](https://www.rdocumentation.org/packages/sampling/versions/2.11/topics/UPmaxentropy))
2. DPPs for diversity sampling — general background on determinantal point processes ([Link](https://dahtah.github.io/Determinantal.jl/dev/))
