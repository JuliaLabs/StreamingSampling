# StreamingSampling.jl

StreamingSampling is a Julia-based proof-of-concept implementation of a streamed variant of maximum-entropy sampling ([UPmaxentropy](https://www.rdocumentation.org/packages/sampling/versions/2.11/topics/UPmaxentropy)). It is designed to process large datasets stored on disk with minimal impact on RAM. The method begins by computing first-order inclusion probabilities using a [DPP](https://dahtah.github.io/Determinantal.jl/dev/)-based heuristic, and then feeds these probabilities into the classical UPmaxentropy algorithm to produce diverse samples.

<a href="https://julia.mit.edu/StreamingSampling.jl/dev/">
<img alt="Development documentation" src="https://img.shields.io/badge/documentation-in%20development-orange?style=flat-square">
</a>
<a href="https://mit-license.org">
<img alt="MIT license" src="https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square">
</a>
<a href="https://github.com/cesmix-mit/PotentialLearning.jl/issues/new">
<img alt="Ask us anything" src="https://img.shields.io/badge/Ask%20us-anything-1abc9c.svg?style=flat-square">
</a>
</a> 
<br />
<br />




