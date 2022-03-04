# Intel Project for LLVM\* technology

This is the Intel staging area for llvm.org contributions and the home for
Intel LLVM-based projects:

- [oneAPI Data Parallel C++ compiler](#oneapi-data-parallel-c-compiler)
- [Late-outline OpenMP and OpenMP Offload](#late-outline-openmp-and-openmp-offload)

## oneAPI Data Parallel C++ compiler

[![](https://spec.oneapi.io/oneapi-logo-white-scaled.jpg)](https://www.oneapi.io/)

[![Linux Post Commit Checks](https://github.com/intel/llvm/workflows/Linux%20Post%20Commit%20Checks/badge.svg)](https://github.com/intel/llvm/actions?query=workflow%3A%22Linux+Post+Commit+Checks%22)
[![Generate Doxygen documentation](https://github.com/intel/llvm/workflows/Generate%20Doxygen%20documentation/badge.svg)](https://github.com/intel/llvm/actions?query=workflow%3A%22Generate+Doxygen+documentation%22)

The Data Parallel C++ or DPC++ is a LLVM-based compiler project that implements
compiler and runtime support for the SYCL\* language. The project is hosted in
the [sycl](/../../tree/sycl) branch and is synced with the tip of the LLVM
upstream main branch on a regular basis (revisions delay is usually not more
than 1-2 weeks). DPC++ compiler takes everything from LLVM upstream as is,
however some modules of LLVM might be not included in the default project build
configuration. Additional modules can be enabled by modifying build framework
settings.

The DPC++ goal is to support the latest SYCL\* standard and work on that is in
progress. DPC++ also implements a number of extensions to the SYCL\* standard,
which can be found in the [sycl/doc/extensions](/../sycl/sycl/doc/extensions)
directory.

The main purpose of this project is open source collaboration on the DPC++
compiler implementation in LLVM across a variety of architectures, prototyping
compiler and runtime library solutions, designing future extensions, and
conducting experiments. As the implementation becomes more mature, we try to
upstream as much DPC++ support to LLVM main branch as possible. See
[SYCL upstreaming working group notes](/../../wiki/SYCL-upstreaming-working-group-meeting-notes)
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
[Containers](/../sycl/sycl/doc/developer/DockerBKMs.md#sycl-containers-overview)

#### Releases

Daily builds of the sycl branch on Linux are available at
[releases](/../../releases).
A few times a year, we publish [Release Notes](/../sycl/sycl/ReleaseNotes.md) to
highlight all important changes made in the project: features implemented and
issues addressed. The corresponding builds can be found using
[search](https://github.com/intel/llvm/releases?q=oneAPI+DPC%2B%2B+Compiler&expanded=true)
in daily releases. None of the branches in the project are stable or rigorously
tested for production quality control, so the quality of these releases is
expected to be similar to the daily releases.

#### Build from sources

See [Get Started Guide](/../sycl/sycl/doc/GetStartedGuide.md).

### Report a problem

Submit an [issue](/../../issues) or initiate a [discussion](/../../discussions).

### How to contribute to DPC++

See [ContributeToDPCPP](/../sycl/sycl/doc/developer/ContributeToDPCPP.md).

## Late-outline OpenMP\* and OpenMP\* Offload

See [openmp](/../../tree/openmp) branch.

# License

See [LICENSE](/../sycl/sycl/LICENSE.TXT) for details.

<sub>\*Other names and brands may be claimed as the property of others.</sub>
