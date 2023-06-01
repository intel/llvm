# SYCL Command Graph Extensions

This is the collaboration space for the oneAPI vendor Command Graph extension for SYCL2020. It provides an API for defining a graph of operations and their dependencies once and submitting this graph repeatedly for execution.

### Specification

A draft of our Command Graph extension proposal can be found here:
[https://github.com/intel/llvm/pull/5626](https://github.com/intel/llvm/pull/5626).

### Implementation

Our current prototype implementation can be found here:
[https://github.com/reble/llvm/tree/sycl-graph-develop](https://github.com/reble/llvm/tree/sycl-graph-develop).

#### Backends

An application can query the SYCL library for the level of support it
provides for using the extension with a device by using
`ext::oneapi::experimental::info::device::graph_support`, which returns one of:

* Native - Backend command-buffer construct is used to implement graphs.
* Emulated - Graphs support is emulated by reissuing commands to the backend.
* Unsupported - Extension is not supported on the device.

Currently the Level Zero backend is the only supported SYCL backend for the
`sycl_ext_oneapi_graph` extension. As the focus of the current prototype is good
Level Zero support to prove the value of the extension, rather than emulated
support for many backends. However, broadening the number of backends supported
is something we are interested in expanding on.

| Backend    | Implementation Support     |
| ---------- | -------------------------- |
| Level Zero | Native using command-lists |
| CUDA       | Unsupported                |
| OpenCL     | Unsupported                |
| HIP        | Unsupported                |
| Others     | Unsupported                |

#### Implementation Status

| Feature                                                            | Implementation Status |
| ------------------------------------------------------------------ | --------------------- |
| Adding a command-group node with `command_graph::add()`            | Implemented           |
| Begin & end queue recording to a graph to create nodes             | Implemented           |
| Edges created from buffer accessor dependencies                    | Implemented           |
| Edges created from `handler::depends_on` dependencies              | Implemented           |
| Edges created using `make_edge()`                                  | Implemented           |
| Edges created by passing a property list to `command_graph::add()` | Implemented           |
| Empty node                                                         | Implemented           |
| Queue `ext_oneapi_get_state()` query                               | Implemented           |
| Vendor test macro                                                  | Implemented           |
| Ability to add a graph as a node of another graph (Sub-graphs)     | Implemented           |
| Using all capabilities of USM in a graph node                      | Implemented           |
| Extending lifetime of buffers used in a graph                      | Not implemented       |
| Buffer taking a copy of host data when buffer is used in a graph   | Not implemented       |
| Executable graph `update()`                                        | Not implemented       |
| Recording an in-order queue preserves linear dependencies          | Not implemented       |
| Using `handler::parallel_for` in a graph node                      | Implemented           |
| Using `handler::single_task` in a graph node                       | Implemented           |
| Using `handler::memcpy` in a graph node                            | Implemented           |
| Using `handler::copy` in a graph node                              | Implemented           |
| Using `handler::host_task` in a graph node                         | Not implemented       |
| Using `handler::fill` in a graph node                              | Implemented for USM, not implemented for buffer accessors |
| Using `handler::memset` in a graph node                            | Not implemented       |
| Using `handler::prefech` in a graph node                           | Not implemented       |
| Using `handler::memadvise` in a graph node                         | Not implemented       |
| Using specialization constants in a graph node                     | Not implemented       |
| Using reductions in a graph node                                   | Not implemented       |
| Using sycl streams in a graph node                                 | Not implemented       |
| Thread safety of new methods                                       | Not implemented       |
| Profiling an event returned from graph submission                  | Not implemented       |
| Querying the state of an event returned from graph submission      | Not implemented       |
| Error checking                                                     | Throwing exceptions for invalid usage is only partially implemented |

### Other Material

This extension was presented at the oneAPI Technical Advisory board (Sept'22 meeting). Slides: [https://github.com/oneapi-src/oneAPI-tab/blob/main/language/presentations/2022-09-28-TAB-SYCL-Graph.pdf](https://github.com/oneapi-src/oneAPI-tab/blob/main/language/presentations/2022-09-28-TAB-SYCL-Graph.pdf).

Extension was presented at IWOCL 2023, and the [talk can be found on Youtube](https://www.youtube.com/watch?v=aOTAmyr04rM).

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
