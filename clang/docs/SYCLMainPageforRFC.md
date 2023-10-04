# Add full support for the SYCL programming model

We (Intel) would like to request to add full support for SYCL programming model support to LLVM/Clang project to facilitate collaboration on C++ single-source heterogeneous programming for accelerators like GPU, FPGA, DSP, etc. from different hardware and software vendors. The SYCL 2020 Specification is available at the Khronos site: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html.

Our prior RFC toward this effort can be seen here: https://lists.llvm.org/pipermail/cfe-dev/2019-January/060811.html

In those four years: 
* both our design and SYCL have evolved (e.g. the previous RFC supported SYCL 1.2.1)
* we have multiple years of SYCL implementation experience
* we have received feedback from the community for both the original RFC as well as patched that we have upstreamed
* we have real-world user feedback from talking with our customers
* we have worked with Codeplay in supporting non-Intel hardware

and as a result now have a more mature set of design and implementation that we can upstream that is more in line with community standards and expectations.

SYCL is an open specification and Intel wants an implementation that fully meets this specification in the community.  This allows for open tooling and is in line with the goals og the Unified Acceleration (UXL) Foundation: https://www.oneapi.io/blog/announcing-the-unified-acceleration-uxl-foundation/

Intel commits to supporting SYCL for the long term.  Alexey Bader, who is deeply involved with both the community and SYCL, is the SYCL code owner.

## Topics of interest

* Driver: Enabling command-line options, adjusting compilation tool chains
* Front End: Integration Header and Footer 
* Front End: Lowering of SYCL Kernel
* Front End: Support for Unnamed Lambdas as SYCL kernels
* Tools: Offloading design for SYCL offload kind and SPIR targets
* Run-time Library: TBD
