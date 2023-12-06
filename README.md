This is the Intel staging area for llvm.org contributions and the home for
Intel LLVM-based projects:

- [oneAPI DPC++ compiler](#oneapi-dpc-compiler)
- [Late-outline OpenMP and OpenMP Offload](#late-outline-openmp-and-openmp-offload)

For general contribution process see [CONTRIBUTING.md](./CONTRIBUTING.md)

## oneAPI DPC++ compiler

[![oneAPI logo](https://spec.oneapi.io/oneapi-logo-white-scaled.jpg)](https://www.oneapi.io/)

[![SYCL Post Commit](https://github.com/intel/llvm/actions/workflows/sycl_post_commit.yml/badge.svg?branch=sycl)](https://github.com/intel/llvm/actions/workflows/sycl_post_commit.yml)
[![Generate Doxygen documentation](https://github.com/intel/llvm/actions/workflows/gh_pages.yml/badge.svg?branch=sycl)](https://github.com/intel/llvm/actions/workflows/gh_pages.yml)

The DPC++ is a LLVM-based compiler project that implements compiler and runtime
support for the SYCL\* language. The project is hosted in the
[sycl](https://github.com/intel/llvm/tree/sycl) branch and is synced with the
tip of the LLVM upstream main branch on a regular basis (revisions delay is
usually not more than 1-2 weeks). DPC++ compiler takes everything from LLVM
upstream as is, however some modules of LLVM might be not included in the
default project build configuration. Additional modules can be enabled by
modifying build framework settings.

The DPC++ goal is to support the latest SYCL\* standard and work on that is in
progress. DPC++ also implements a number of extensions to the SYCL\* standard,
which can be found in the [sycl/doc/extensions](./sycl/doc/extensions)
directory.

The main purpose of this project is open source collaboration on the DPC++
compiler implementation in LLVM across a variety of architectures, prototyping
compiler and runtime library solutions, designing future extensions, and
conducting experiments. As the implementation becomes more mature, we try to
upstream as much DPC++ support to LLVM main branch as possible. See
[SYCL upstreaming working group notes](https://github.com/intel/llvm/wiki/SYCL-upstreaming-working-group-meeting-notes)
for more details.

Note that this project can be used as a technical foundation for some
proprietary compiler products, which may leverage implementations from this open
source project. One of the examples is
[Intel(R) oneAPI DPC++ Compiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/dpc-compiler.html)
Features parity between this project and downstream projects is not guaranteed.

Project documentation is available at:
[DPC++ Documentation](https://intel.github.io/llvm-docs/).

### How to use DPC++

#### Docker containers

See available containers with pre-built/pre-installed DPC++ compiler at:
[Containers](./sycl/doc/developer/DockerBKMs.md#sycl-containers-overview)

#### Releases

Daily builds of the sycl branch on Linux are available at
[releases](https://github.com/intel/llvm/releases).
A few times a year, we publish [Release Notes](./sycl/ReleaseNotes.md) to
highlight all important changes made in the project: features implemented and
issues addressed. The corresponding builds can be found using
[search](https://github.com/intel/llvm/releases?q=oneAPI+DPC%2B%2B+Compiler&expanded=true)
in daily releases. None of the branches in the project are stable or rigorously
tested for production quality control, so the quality of these releases is
expected to be similar to the daily releases.

#### Build from sources

See [Get Started Guide](./sycl/doc/GetStartedGuide.md).

### Report a problem

Submit an [issue](https://github.com/intel/llvm/issues) or initiate a 
[discussion](https://github.com/intel/llvm/discussions).

### How to contribute to DPC++

This project welcomes contributions from the community. Please refer to [CONTRIBUTING](/CONTRIBUTING.md) 
for general guidelines around contributing to this project. You can then see 
[ContributeToDPCPP](./sycl/doc/developer/ContributeToDPCPP.md) for DPC++ specific 
guidelines.

## Late-outline OpenMP\* and OpenMP\* Offload

See [openmp](/openmp) branch.

# License

See [LICENSE](./sycl/LICENSE.TXT) for details.

<sub>\*Other names and brands may be claimed as the property of others.</sub>
