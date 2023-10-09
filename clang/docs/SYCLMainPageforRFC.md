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

## Purpose of this RFC

The purpose of this RFC is twofold.  Much of the RFC describes how we propose to support SYCL in clang in the near term using our current work.  However, we would also like to describe the limitations we have found and outline a direction for how these limitations could be lifted.  We are particularly interested in the community’s thoughts on these directions.  We are, of course, committed to improving the design as we maintain the SYCL support.

### Support for third party host compilers

Like many offload projects in clang, the design for SYCL takes a multi-pass compiler approach.  The application’s source file is compiled once with a “device compiler”, which mostly ignores the host code and generates code only for the offload kernels.  The source file is then compiled a second time with a “host compiler”, which generates code only for the host parts of the application.  Normally, the host and device compilers are both clang, but run with different options.  Some users, however, need to use their own host compiler, which might not be clang.  Our design currently supports this through a command line option named “-fsycl-host-compiler”.

Our preference is to upstream this support also because we think it is generally useful for people using clang to compile SYCL programs.  Other clang developers may also fork clang as a starting point for their own SYCL implementations, and they could also benefit from the common support for a third-party host compiler.  Does the community see this as sufficient reason to upstream support that interoperates with a non-clang compiler?

### Mechanism for integrating the host and device code

One central problem that must be solved with a multi-pass compiler design is to logically connect the host and device code, which are separately compiled.  Our current strategy is to compile the device code first, and have the device compiler generate two header files which are automatically included into the host code.  One header (the integration header) is automatically included at the start of the host compilation phase and the other header (the integration footer) is automatically included at the end of the host compilation phase.  This design is described in detail in https://github.com/intel/llvm/pull/11431.

This design is well suited for supporting a third-party host compiler because the connection between the device and host code is standard C++ header code, which can be consumed by any conformant C++ host compiler.  However, we have observed that relying on standard C++ imposes certain limitations on the input SYCL source code.

One example is the use of the C++ type that is used to “name” a kernel.  With the header file approach, these types must be forward declared in the integration header.  However, this places a limitation on these types to those that can be legally forward declared at namespace scope.  For example, it is not possible to forward declare a type that is defined at block scope or to forward declare a type that is in the “std” namespace.  Thus, these types may not be legally used to name kernels in the user’s SYCL source code.

We would like to lift these limitations, at least in the common case where the user uses clang for both the host and device compilation phases.  Our thought was to implement a different integration approach for this common case, which might assume that the host compiler understands clang attributes like __builtin_sycl_unique_stable_name and decorates the LLVM IR with some new attributes.  Our thinking is very early at this point.

Even if we develop this new approach, though, we think it makes sense to keep the approach based on the integration headers because this will continue to be used for a third-party host compiler.

### Location for the logic for generating the integration support

Another open question is the location for the logic that generates the integration headers (or the new integration information alluded to above).  This logic is currently located in the sema phase of the CFE, and the community has expressed concern about this.  We are considering two alternatives.  One option is to move the logic to the codegen phase of the CFE.  Another is to move the logic out of the CFE entirely and create a new LLVM IR pass instead.  If we do this, the new IR pass would likely run near the start of the pipeline, before other IR passes can transform the IR that is emitted by the CFE.  This option may be attractive especially if we end up supporting two integration methods because we could isolate each method in its own IR pass, and then enable one or the other pass depending on whether the user requests a third-party host compiler.

### Dependency on the Khronos LLVMIR / SPIRV translator

With our current approach, a SYCL source file is compiled to a fat object, which contains both host code and embedded LLVM IR for the device code.  When these objects are linked together, the LLVM IR modules from all objects are linked together, and then the sycl-post-link tool performs some final IR transformations that require visibility to the entire IR for each offload kernel.  After these final IR passes complete, we translate the LLVM IR to SPIR-V, which is embedded in the fat executable.

This final translation to SPIR-V currently relies on the Khronos SPIR-V translator, which is an external project.  As a result, the end user must ensure that the translator is installed and on the PATH in order have a working clang for SYCL.  We would like to eliminate this dependency in order to make it easier for people to use clang for SYCL.

We are investigating ways to use the clang SPIR-V backend in place of the Khronos translator by using it at the end of the sycl-post-link tool.

## Topics of interest

In this RFC, we would like to request your feedback for the following items:

* Driver: Enabling command-line options, adjusting compilation tool chains
* Front End: Integration Header and Footer 
* Front End: Lowering of SYCL Kernel
* Front End: Support for unnamed SYCL kernel functions
* Tools: Offloading design for SYCL offload kind and SPIR targets
* Run-time Library: TBD
