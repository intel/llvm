# Offloading design for SYCL offload kind and SPIR targets

This RFC is intended to discuss proposed changes to compilation flow for
offloading SYCL kernels specifically to SPIR-based targets. Most of the changes
will be made in the `clang-linker-wrapper` tool. To be posted to the Clang
Frontend category.

## Introduction

Traditional device offloading models are completely encapsulated within the
compiler driver requiring the driver to perform all of the steps required for
generating the host and device compilation passes. The driver is also
responsible for initiating any of the link-time processing that occurs for each
device target.

An updated offloading model uses the new `clang-linker-wrapper` tool. Much of the
functionality that is performed during the link phase of the offloading
compilation is removed from the driver and moved to the `clang-linker-wrapper`
tool.

Below is a general representation of the overall offloading flow that is
performed during a full compilation from source to final executable. The
compiler driver is responsible for creating the fat object and the
`clang-linker-wrapper` tool is responsible for the general functionality that is
performed during the link.

![High level view of the offloading flow](images/offloadflow.svg)

*Diagram 1: Overall compilation flow*

## Fat object generation for SYCL offload kinds using `clang-offload-packager`

`clang-offload-packager` plays a vital role during fat-object generation. The fat
object in the proposed offloading model is generated during the host
compilation. The host compilation takes an additional argument which points to
the device binary which will be embedded in the final object. Generation will
be separated out to allow for potential parallelism during compilation of both
the host and target device binaries.

When dealing with multiple device binaries, an additional step is performed to
package the multiple device binaries before being added to the host object.
This additional step is performed with the `clang-offload-packager` taking image
inputs containing information relating to the target triple, architecture
setting and offloading kind.

The `clang-offload-packager` is run during \'fat object\' generation regardless
of the number of device binaries being added to the conglomerate fat object.
The device binaries are contained in what is designated as an 'Offload Binary'.
These binaries can reside in a variety of binary formats including Bitcode
files, ELF objects, executables and shared objects, COFF objects, archives or
simply stored as an offload binary.

We should have the ability to package SPIR-V based device binaries in the
offload section of any given binary. These device binaries will be packaged as
normal with the packager and placed within the given section.

Example usage of the external `clang-offload-packager` call:

`clang-offload-packager --image=file=<name>,triple=<triple>,kind=<kind>`

In the proposed offloading model, the compiler driver is responsible for
creating the fat object. There are two options to generate the fat object:
- Device code embedded in the fat object is LLVM IR code
- Device code embedded in the fat object is SPIR-V IR code (-fsycl-rdc)
- Device code embedded in the fat object is SPIR-V IR code (-fno-sycl-rdc)

Table below showcases the pros and cons of each approach

|            Choice                    |   Pros                       |   Cons                      |
|--------------------------------------|------------------------------|-----------------------------|
| Device code embedded in the fat object is LLVM IR code | Well tested post-link passes that work on LLVM IR are available and can be upstreamed | Cross-release compatibility will be a greater challenge to handle as
LLVM Spec is more fluid that SPIR-V Spec |
| 2a. Device code embedded in the fat object is SPIR-V IR code (-fsycl-rdc) | Less effort to maintain cross-release compatibility | Additional passes required to translate SPIR-V to LLVM to facilitate running of post-link passes on LLVM IR and then translate LLVM IR back to SPIR-V. There might be loss of vital metadata in this process |
| 2b. Device code embedded in the fat object is SPIR-V IR code (-fno-sycl-rdc). Less effort to maintain cross-release compatibility | In this scenario, there is no linking necessary and hence all post-link operations can be done early and then the result can be translated to SPIR-V | Cross-module linking is not available |

*Table: Pros and cons to evaluate design choice to decide whether fat object should contain LLVM IR or SPIR-V IR*

`clang-offload-packager` will be used to embed device code into the host code.
Following changes will be added to the packager. A new offload kind (SYCL_OFK)
will be made available for SYCL offloads. We should have the ability to package
SPIR-V based device binaries in the offload section of any given binary. These
device binaries will be packaged as normal with the packager and placed within
the given section. New image kinds will be added to represent such binaries. 

## SYCL offload support in `clang-linker-wrapper`

The `clang-linker-wrapper` provides the interface to perform the needed link
steps when consuming fat binaries. The linker wrapper performs a majority of
the work involved during the link step during an offload compilation,
significantly reducing the amount of work that is occurring in the compiler
driver. From the compilation perspective, the linker wrapper replaces the
typical call to the host link. This allows for the responsibility of the
compiler driver to be nearly identical when performing a regular compilation
vs an offloading compilation.

From a high level, using the `clang-linker-wrapper` provides following benefits:
- Moves all of the device linking responsibility out of the compiler driver
- Allows for a more direct ability to perform linking for offloading without
requiring the use of the driver, using more linker like calls
- Provides additional flexibility with the ability to dynamically modify the
toolchain execution.

Example usage of the external `clang-linker-wrapper` call:

`clang-linker-wrapper <wrapper opts> -- <linker opts>`

Following sub-sections cover the different compilation steps invoked inside the
`clang-linker-wrapper`. Changes needed to add SYCL compilation support is
showcased in each sub-section.

### Device code extraction and linking

During the compilation step, the device binaries are embedded in a section of
the host binary. When performing the link, this section is extracted from the
object and mapped according to the device kind. The `clang-linker-wrapper` is
responsible for examining all of the input binaries, grabbing the embedded
device binaries and determining any additional device linking paths that need
to be taken.

A new device offload kind is made available for SYCL offloads. New device image
kinds will be added to represent SPIR-V code and AOTcompiled device code. All
input bitcode files will be linked together using the ThinLTO pass. In
addition, SYCL device library files will be provided as inputs by the driver
and will be linked with the input. A list of device libraries that need to
be linked in with user code is provided by the driver. The driver is also
responsible for letting the `clang-linker-wrapper` know the location of the
device libraries.  

|            Option                    |   Expected Behavior   |
|--------------------------------------|-----------------------|
| `--sycl-device-libraries=<arg>`      | A comma separated list of device libraries that are linked during the device link |
| `--sycl-device-library-location=<arg>`    | The location in which the device libraries reside |

*Table: Options to pass device libraries to the `clang-linker-wrapper`*

### Post-link and SPIR-V translation

After the device binaries are linked together, two additional steps are
performed to prepare the device binary for consumption by an offline
compilation tool for AOT or to be wrapped for JIT processing.

The `sycl-post-link` tool is used after the device link is performed, applying
any changes such as optimizations and code splitting before passing off to the
`llvm-spirv` tool, which translates the LLVM-IR to SPIR-V.

|            Option                    |   Expected Behavior   |
|--------------------------------------|-----------------------|
| `--sycl-post-link-options=<arg>`     | Options that will control `sycl-post-link` step |
| `--llvm-spirv-options=<arg>`         | Options that will control `llvm-spirv` step |

*Table: Options to pass `sycl-post-link` and `llvm-spirv` options to the `clang-linker-wrapper`*

Options that will be used by clang-linker-wrapper when invoking the `sycl-post-link`
tool are provided by the driver via the `--sycl-post-link-options=<arg>` option.
Options that will be used by clang-linker-wrapper when invoking the `llvm-spirv`
tool are provided by the driver via the `--llvm-spirv-options=<arg>` option.

### Ahead of Time Compilation for SYCL offload

The updated offloading model will integrate the Ahead of Time (AOT) compilation
behaviors into the `clang-linker-wrapper`. The actions will typically take place
after the device link, post link, and LLVM-IR to SPIR-V translation steps.

Regardless of the AOT target, the flow is similar, only modifying the offline
compiler that is used to create the target device image. It is expected that
the offline compiler will also use unique command lines specific to the tool to
create the image.

To support the needed option passing triggered by use of the
`-Xsycl-target-backend` option and implied options based on the optional device
behaviors for AOT compilations for GPU new command line interfaces are needed
to pass along this information.

| Target | Triple        | Offline Tool   | Option for Additional Args |
|--------|---------------|----------------|----------------------------|
| CPU    | spir64_x86_64 | opencl-aot     | `--cpu-tool-arg=<arg>`     |
| GPU    | spir64_gen    | ocloc          | `--gen-tool-arg=<arg>`     |
| FPGA   | spir64_fpga   | aoc/opencl-aot | `--fpga-tool-arg=<arg>`    |

*Table: Ahead of Time Info*

To complete the support needed for the various targets using the
clang-linker-wrapper as the main interface, a few additional options will be
needed to communicate from the driver to the tool. Further details of usage are
given below.

| Option Name                  | Purpose                                      |
|------------------------------|----------------------------------------------|
| `--fpga-link-type=<arg>`     | Tells the link step to perform 'early' or 'image' processing to create archives for FPGA |
| `--parallel-link-sycl=<arg>` | Provide the number of parallel jobs that will be used when processing split jobs |

*Table: Additional Options for clang-linker-wrapper*

The `clang-linker-wrapper` provides an existing option named `-wrapper-jobs`
that may be useful for our usage.

#### spir64_gen support

Compilation behaviors involving AOT for GPU involve an additional call to
the OpenCL Offline compiler (OCLOC).  This call occurs after the post-link
step performed by `sycl-post-link` and the SPIR-V translation step which is done
by `llvm-spirv`.  Additional options passed by the user via the
`-Xsycl-target-backend=spir64_gen <opts>` command as well as the implied
options set via target options such as `-fsycl-targets=intel_gpu_skl`
will be processed by a new options to the wrapper, `--gen-tool-arg=<arg>`

To support multiple target specifications, for instance:
`-fsycl-targets=intel_gpu_skl,intel_gpu_pvc`, multiple `--gen-tool-arg`
options can be passed on the command line.  Each instance will be considered
a separate OCLOC call passing along the `<args>` as options to the OCLOC call.
The compiler driver will be responsible for putting together the full option
list to be passed along.

> -fsycl -fsycl-targets=spir64_gen,intel_gpu_skl
-Xsycl-target-backend=spir64_gen "-device pvc -options -extraopt_pvc"
-Xsycl-target-backend=intel_gpu_skl "-options -extraopt_skl"

*Example: spir64_gen enabling options*

> --gen-tool-arg="-device pvc -options extraopt_pvc"
--gen-tool-arg="-device skl -options -extraopt_skl"

*Example: `clang-linker-wrapper` options*

Each OCLOC call will be represented as a separate device binary that is
individually wrapped and linked into the final executable.

Additionally, the syntax can be expanded to enable the ability to pass specific
options to a specific device GPU target for spir64_gen.  The syntax will
resemble `--gen-tool-arg=<arch> <arg>`.  This corresponds to the existing
option syntax of `-fsycl-targets=intel_gpu_arch` where `arch` can be a fixed
set of targets.

#### spir64_x86_64 support

Compilation behaviors involving AOT for CPU involve an additional call to
`opencl-aot`.  This call occurs after the post-link step performed by
`sycl-post-link` and the SPIR-V translation step performed by `llvm-spirv`.
Additional options passed by the user via the
`-Xsycl-target-backend=spir64_x86_64 <opts>` command will be processed by a new
option to the wrapper, `--cpu-tool-arg=<arg>`

### Wrapping of device images

Once the device binary is pulled out of the fat binary, the binary must be
wrapped and provided the needed entry points to be used during execution.
This is performed during the link phase and controlled by the
`clang-linker-wrapper`.

SYCL offload model currently uses specialized wrapping information to wrap
device images into host. It is expected that the wrap information that will be
generated in `clang-linker-wrapper` to be wrapped around the device binary will
match wrapping information that is used for SYCL.

### Host link

The final host link is also performed by the linker wrapper. This link is built
upon the full link command line as constructed by the compiler driver, including
all libraries and the linked/wrapped device binaries to complete the compilation
process. We do not expect any changes in this step.