# SYCL Command Graph Extensions

This is the collaboration space for the oneAPI vendor Command Graph extension for SYCL2020. It provides an API for defining a graph of operations and their dependencies once and submitting this graph repeatedly for execution.

### Specification

A draft of our Command Graph extension proposal can be found here:
[https://github.com/intel/llvm/pull/5626](https://github.com/intel/llvm/pull/5626).

### Implementation

Our current prototype implementation can be found here:
[https://github.com/reble/llvm/tree/sycl-graph-develop](https://github.com/reble/llvm/tree/sycl-graph-develop).

Limitations include:
* LevelZero backend support only. A fallback emulation mode is used for correctness on other backends.
* Accessors and reductions are currently not supported.

### Other Material

This extension was presented at the oneAPI Technical Advisory board (Sept'22 meeting). Slides: [https://github.com/oneapi-src/oneAPI-tab/blob/main/language/presentations/2022-09-28-TAB-SYCL-Graph.pdf](https://github.com/oneapi-src/oneAPI-tab/blob/main/language/presentations/2022-09-28-TAB-SYCL-Graph.pdf).

## Intel Project for LLVM\* technology

We target a contribution through the origin of this fork: [Intel staging area for llvm.org contributions](https://github.com/intel/llvm).

### How to use DPC++

#### Releases

TDB

#### Build from sources

See [Get Started Guide](./sycl/doc/GetStartedGuide.md).

### Report a problem

Submit an [issue](https://github.com/intel/llvm/issues) or initiate a 
[discussion](https://github.com/intel/llvm/discussions).

### How to contribute to DPC++

See [ContributeToDPCPP](./sycl/doc/developer/ContributeToDPCPP.md).

# License

See [LICENSE](./sycl/LICENSE.TXT) for details.

<sub>\*Other names and brands may be claimed as the property of others.</sub>
