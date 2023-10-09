# Add full support for the SYCL programming model

We (Intel) propose adding full support for the SYCL programming model support to the LLVM/Clang project to facilitate collaboration on C++ single-source heterogeneous programming for accelerators (e.g., GPU, FPGA, DSP) from different hardware and software vendors. The SYCL 2020 Specification is available at the Khronos site: https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html.

Our prior RFC toward this effort can be seen here: https://lists.llvm.org/pipermail/cfe-dev/2019-January/060811.html

In the four years since then: 
  *   both our design and SYCL have evolved (e.g. the previous RFC supported SYCL 1.2.1)
  *   we have multiple years of SYCL implementation experience
  *   we have received feedback from the community for both the original RFC as well as patches that we have upstreamed
  *   we have real-world user feedback from talking with our customers
  *   we have worked with Codeplay in supporting non-Intel hardware
and as a result now have a more mature set of design and implementation that we can upstream that is more in line with community standards and expectations.

SYCL is an open specification from the Khronos Group (https://www.khronos.org/sycl/) and Intel supports a community implementation that fully meets this specification.  This would allow for open tooling and be in line with the goals of the Unified Acceleration (UXL) Foundation: https://www.oneapi.io/blog/announcing-the-unified-acceleration-uxl-foundation/

We have a long term commitment to supporting SYCL.  Alexey Bader is the SYCL code owner and has a long history of work within the LLVM community and the development of the SYCL specification.  We have worked with Codeplay to enable non-Intel hardware and hope that a community implementation will further enable SYCL support on a broader range of devices.

## Topics of interest

In this RFC, we would like to request your feedback for the following items:

* Driver: Enabling command-line options, adjusting compilation tool chains
* Front End: Integration Header and Footer 
* Front End: Lowering of SYCL Kernel
* Front End: Support for unnamed SYCL kernel functions
* Tools: Offloading design for SYCL offload kind and SPIR targets
* Run-time Library: TBD
