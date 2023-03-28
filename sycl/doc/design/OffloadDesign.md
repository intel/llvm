# Implementation design for offloading model

## Introduction

This document covers the implementation design for using the new offloading
model for the DPC++ Compiler.  This leverages the existing community Offloading 
design [OffloadingDesign][1] which covers the Clang driver and code generation
steps for creating offloading applications.

[1]: <../../../clang/docs/OffloadingDesign.rst>

The current offloading model is completely encapsulated within the Clang
Compiler Driver requiring the driver to perform all of the additional steps
for generating the host and device compilation passes.  The Driver is also
responsible for performing any of the link-time processing that occurs for
each device target.

The updated offloading model removes much of the functionality that is
performed during the link phase of the offloading compilation and moves it
to a `clang-linker-wrapper` tool.

Below is a general representation of the overall offloading flow that is
performed during a full compilation from source to final executable.  The
compiler driver is responsible for creating the fat object and the
`clang-linker-wrapper` tool is responsible for the general functionality that is
performed during the link.

![High level view of the offloading flow](images/OffloadGeneralFlow.png
 "General Offload Flow")

*Diagram 1: General Offload Flow*

More specific details of the clang-linker-wrapper behavior can be seen in the
[clang-linker-wrapper](#clang-linker-wrapper) section.

## Fat Binary Representation

Binaries generated during the offload compilation will be 'bundled' together
to create a conglomerate fat binary.  Depending on the type of binary, the
packaging of the device section will be different.

## Device only Binaries

Generation and consumption of device only binaries is also available.

## Fat Binary Generation

The generation of the fat binary will take place in the driver.  The model from
the community is generating the fat binary as a secondary process when creating
the host object.  Generation will be separated from the host compilation step.
This is being done to enable proper support for using an external host compiler
as well as taking advantage of potential parallelism during compilation of both
the host and target device binaries.

The fat object in the new model is generated during the host compilation.
The host compilation takes an additional argument which points to the device
binary which will be embedded in the final object.  Generation will be
separated out to allow for potential parallelism during compilation of both the
host and target device binaries.

![Creating the fat object](images/OffloadFatObject.png
 "Fat Object Generation")

*Diagram 2: Fat Object Generation*

### Packager

When dealing with multiple device binaries, an additional step is performed to
package the multiple device binaries before being added to the host object.
This additional step is performed with the `clang-offload-packager` taking
image inputs containing information relating to the target triple,
architecture setting and offloading kind.

The `clang-offload-packager` is run during 'fat object' generation regardless
of the number of device binaries being added to the conglomerate fat object.
The device binaries are contained in what is designated as an ‘Offload Binary’.
These binaries can reside in a variety of binary formats including Bitcode
files, ELF objects, executables and shared objects, COFF objects, archives or
simply stored as an offload binary.

We should have the ability to package SPIR-V based device binaries in the
offload section of any given binary.  These device binaries will be packaged
as normal with the packager and placed within the given section.

## Clang Linker Wrapper

The clang-linker-wrapper provides the interface to perform the needed link
steps when consuming fat binaries.  The linker wrapper performs a majority of
the work involved during the link step during an offload compilation,
significantly reducing the amount of work that is occuring in the compiler
driver.  From the compilation perspective, the linker wrapper replaces the
typical call to the host link.  This allows for the responsibility of the
compiler driver to be nearly identical when performing a regular compilation
vs an offloading compilation.

Example usage of the external `clang-linker-wrapper` call:

`clang-linker-wrapper <wrapper opts> -- <linker opts>`

### Device Extraction

During the compilation step, the device binaries are embedded in a section of
the host binary. When performing the link, this section is extracted from the
object and mapped according to the device kind.

Depending on the type of binary, the device is embedded as follows:

#### Objects

Object types considered are COFF objects, ELF executables, ELF relocatable
objects and ELF shared objects.  The device section of the object is in a
section marked by `.llvm.offloading` for COFF objects.  For ELF files, there is
a section marked with the `LLVM_OFFLOADING` type.

#### Bitcode

The section which contains the offloading data is from the `.llvm.offloading`
section within the `llvm.embedded.object` metadata.

#### Offload Binary

The binary itself can be represented by just an offload binary, not requiring
to be in a section of another binary.  This representation is used for any
kind of device only binary that is created.

#### Archives

Each item in the archive will be extracted and will be individually checked for
the file type, properly performing extraction based on the file types already
listed above.

### Device Linking

During this phase, all of the individual device binaries that are extracted and
are associated with a given target are worked on given the type of binary we
are working with.  The default device code is typically represented in LLVM-IR
which requires an additional link step of the device code before being
wrapped and integrated into the final executable.  As mentioned in
[Packager](#packager) device representation in SPIR-V should be considered
with the ability to link native SPIR-V modules.

To reduce the potential size of the linked device binary, an additional host
link step is performed to gather dependency information when static device
libraries are being compiled.  This information is sent through the
`clang-offload-deps` tool to generate a dependency IR file which is used
during the device link step.

The use of the clang-linker-wrapper introduces the support of LTO for
device code.  We can leverage this and move away from the dependency gathering
information step with 'clang-offload-deps' and use thinLTO for device code.

There are multiple device linking steps that also occur.  The first step links
together all of the objects and the required device libraries.  The second is
performed including all of optional device libraries, the static device
libraries and the dependency information that was gathered above.  This link
step is performed with `--only-needed` to streamline the final device binary.

The device libraries that are pulled in during the device link step will need
to be controlled by the `clang-linker-wrapper`.  There is a controlling option
(`-fno-sycl-device-lib=arg`) that is available to the user which needs to be
provided to the wrapper to give more control over what device libraries are
pulled in during the device link.

### Ahead Of Time Compilation

The updated offloading model will integrate the Ahead of Time (AOT) compilation
behaviors into the clang-linker-wrapper.  The actions will typically take place
after the device link and post link steps.

Regardless of the AOT target, the flow is similar, only modifying the offline
compiler that is used to create the target device image.  It is expected that
the offline compiler will also use unique command lines specific to the tool
to create the image.

To support the needed option passing triggered by use of the
`-Xsycl-target-backend` option and implied options based on the optional
device behaviors for AOT compilations for GPU new command line interfaces
are needed to pass along this information.

| Target | Triple        | Offline Tool   | Option for Additional Args |
|--------|---------------|----------------|----------------------------|
| CPU    | spir64_x86_64 | opencl-aot     | `--cpu-tool-arg=<arg>`     |
| GPU    | spir64_gen    | ocloc          | `--gen-tool-arg=<arg>`     |
| FPGA   | spir64_fpga   | aoc/opencl-aot | `--fpga-tool-arg=<arg>`    |

*Table: Ahead of Time Info*

To complete the support needed for the various targets using the
`clang-linker-wrapper` as the main interface, a few additional options will
be needed to communicate from the driver to the tool.  Further details of usage
are given further below.

| Option Name                  | Purpose                                      |
|------------------------------|----------------------------------------------|
| `--fpga-tool-deps=<arg>`     | Comma separated list of dependency files used for FPGA hardware compiles using `aoc` |
| `--parallel-link-sycl=<arg>` | Provide the number of parallel jobs that will be used when processing with `llvm-foreach` |
| `--no-sycl-device-lib=<arg>` | Provide the list of device libraries to restrict from linking during device link |

*Table: Additional Options for clang-linker-wrapper*

The `clang-linker-wrapper` provides an existing option named `-wrapper-jobs`
that may be useful for our usage instead of creating a new option specific
to `llvm-foreach` processing.

#### spir64_gen support

Compilation behaviors involving AOT for GPU involve an additional call to
the OpenCL Offline compiler (OCLOC).  This call occurs after the post-link
step performed by sycl-post-link.  Additional options passed by the user
via the `-Xsycl-target-backend=spir64_gen <opts>` command as well as the
implied options set via target options such as `-fsycl-targets=intel_gpu_skl`
will be processed by a new options to the wrapper, `--gen-tool-arg=<arg>`

#### spir64_fpga support

Compilation behaviors involving AOT for FPGA involve an additional call to
the either `aoc` (for Hardware) or `opencl-aot` (for Simulation).  This call
occurs after the post-link step performed by sycl-post-link.  Additional
options passed by the user via the `-Xsycl-target-backend=spir64_fpga <opts>`
command will be processed by a new options to the wrapper,
`--fpga-tool-arg=<arg>`

The FPGA target also has support for additional generated binaries that
contain intermediate files specific for FPGA.  These binaries (aoco, aocr and
aocx) can reside in archives and are treated differently than traditional
device binaries.

When using the `-fintelfpga` option to enable AOT for FPGA, there are
additional expectations during the compilation.  Use of the option will enable
debug generation and also generate dependency information.  The dependency
generation should be packaged along with the device binary for use during
the link phase.  If we are not generating an object and performing compilation
directly to the final executable, a new option named `--fpga-tool-deps=<arg>`
will be needed to pass along the name of the dependency files created during
the compile.  The dependency information is only used when compiling for
hardware.

#### spir64_x86_64 support

Compilation behaviors involving AOT for CPU involve an additional call to
`opencl-aot`.  This call occurs after the post-link step performed by
`sycl-post-link`.  Additional options passed by the user via the
`-Xsycl-target-backend=spir64_x86_64 <opts>` command will be processed by a new
option to the wrapper, `--cpu-tool-arg=<arg>`

### Integration of the sycl-aspect-filter

### Wrapping of device image

Once the device binary is pulled out of the fat binary, the binary must be
wrapped and provided the needed entry points to be used during execution.  This
is performed by the during the link phase and controlled by the
`clang-linker-wrapper`.

It is expected that the wrap information that is generated to be wrapped
around the device binary will match current wrapping information that is used
for the exiting offload model.  The wrapping in the old model is using the
`clang-offload-wrapper` tool.

### Integration of llvm-foreach

Use of `llvm-foreach` is used frequently during the offloading process.  The
functionality will persist with the new model.  Steps using `llvm-foreach`
have the capability of kicking of the steps in a parallel manner by using the
`-fsycl-max-parallel-link-jobs` option on the command line.  Information
about the parallel jobs will need to be passed through the
`clang-linker-wrapper` to be properly processed.  The option will be named
`--parallel-link-sycl=<arg>` to be consumed and used during `llvm-foreach`
toolchain events.

#### Beyond llvm-foreach and similar job hiding tools

Tools like `llvm-foreach`, `file-table-tform`, `spirv-to-ir-wrapper` were all
introduced to provide a way to manipulate behaviors that could only be
determined at runtime of the compiler toolchain.  These were needed to work
around the fact that the toolchain commands constructed by the driver is a fixed
state of commands.

Moving the functionality into `clang-linker-wrapper` presents the opportunity
step away from the static command construction and create the call chain on
the fly based on real time output from corresponding tools being called.

### Host Link

The final host link is also performed by the linker wrapper.  This link is
built upon the full link command line as constructed by the compiler driver,
including all libraries and the linked/wrapped device binaries to complete the
compilation process.

The provided command line for the final host link step contains the full list
of libraries and objects to be linked against.  The expectation is for this
list to be complete.  With the old model, the host objects are directly passed
to the host link step.  The device objects are processed separately.  As we are
passing the full command line to the link step, the objects provided will need
to be full fat objects.  This is different from the old model which will
require for an additional step before the link to create the full fat object
that is properly represented on the host link command line.  This additional
step is necessary due the fact that we are creating the fat objects during
a separate step as opposed to integrating the offload binaries during the
host object generation.  See [Fat Binary Generation](#fat-binary-generation).

## Transitioning from old model to new model

The binary representation of the fat objects is not equivalent when dealing
with differences between the old and the new model.  Behavior of the new
model will be guarded by the `--offload-new-driver` compiler switch.  This will
allow for implementation of the model without disturbing the existing behavior.
When we are ready to make the switch over, it is a matter of making the
switch the default mode.
