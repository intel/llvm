# [RFC] SYCLBIN - A format for SYCL device code

## Summary

This RFC proposes the addition of SYCLBIN, a new binary format for storing SYCL device code. The format provides a lightweight, extensible wrapper around device modules and their corresponding SYCL-specific metadata to be produced/consumed by tools/SYCL runtime.

## Purpose of this RFC

This RFC seeks community feedback on the proposed SYCLBIN binary format design, including:

- The binary format specification and layout
- Toolchain integration approach and new compiler flags

Community input is particularly valuable regarding potential integration challenges with existing LLVM offloading implementations.

## Motivation and alternatives considered

### Requirements: SYCL-Specific metadata and modules hierarchy

The SYCL programming model requires device images to be accompanied by specific metadata necessary for SYCL runtime operation:

1. Device target triple (e.g. ``spirv64_unknown_unknown``).
2. Compiler and linker options for JIT compilation scenarios
3. List of entry points exposed by each image
4. Arrays of [property sets](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/PropertySets.md).

When a binary contains multiple images, some may share common metadata. Therefore, we require a hierarchical structure that enables metadata sharing while allowing specification of image-specific metadata.

#### Existing Format Limitations:

LLVM's offloading infrastructure supports several binary formats that can be embedded within the OffloadBinary format. However, these formats have various limitations that make them unsuitable for SYCL:

- **Single-module design**: Formats like Object, Bitcode, CUBIN, PTX, and SPIRV are designed for single binary or single-module IR representation, lacking hierarchical structuring capabilities for multiple images/modules
- **Missing SYCL metadata support**: None provide native support for SYCL-specific metadata requirements
- **Vendor constraints**: Fatbinary is NVIDIA proprietary and incompatible with SYCL's vendor-neutral approach
- **Limited container capabilities**: OffloadBinary is not designed for multiple device images or hierarchical organization, with StringData insufficient for complex metadata structures (like #3 and #4 above).

### Abstraction: Simplifying Support in Offloading Tools

Another motivation for adding the SYCLBIN format is to encapsulate SYCL-specific logic within SYCL-specific toolchain components (clang-sycl-linker, SYCL runtime) and isolate SYCL implementation details from general offloading tools designed to support multiple programming models.

#### Current Workflow Without SYCLBIN:

Without this format, metadata transfer from compiler to runtime requires the following complex workflow:

1. **clang-sycl-linker** uses OffloadingImage's StringData (with workarounds) to store metadata items #1-#4
2. **clang-linker-wrapper** opens OffloadingImage files from clang-sycl-linker and generates device image binary descriptors in a format readable by SYCL runtime
- Problem: This requires clang-linker-wrapper to maintain SYCL-specific format knowledge, creating unnecessary code duplication
3. **SYCL runtime** decodes metadata using this intermediate format

#### Simplified Workflow With SYCLBIN:

The SYCLBIN format enables a cleaner separation of concerns:

1. **clang-sycl-linker** prepares a complete SYCLBIN containing all metadata and multiple images, embedding it as a single image within OffloadingImage
2. **clang-linker-wrapper** generates only host register/unregister calls without needing knowledge of SYCLBIN internals
3. **SYCL runtime** works directly with SYCLBIN format

This approach eliminates the need for clang-linker-wrapper to understand SYCL-specific formats, reducing maintenance burden and improving toolchain modularity.

### Enable modular dynamic loading of device binaries at runtime

Some applications require dynamic loading of device binaries at runtime to achieve modularity and avoid recompiling the entire application when device code changes. The SYCLBIN format provides a standardized interface between compiler-produced binaries and runtime handling, enabling efficient dynamic loading scenarios.

## Design

### SYCLBIN binary format

The SYCLBIN format consists of:

- A [file header](#file-header) with magic number (0x53594249 "SYBI") and version information
- Three lists of headers: the [abstract module header](#abstract-module-header) list, the
[IR module header](#ir-module-header) list and
[native device code image header](#native-device-code-image-header) list,
containing information about the [abstract modules](#abstract-module),
[IR modules](#ir-module) and
[native device code images](#native-device-code-image) respectively.
- Two byte tables containing metadata and binary data

#### File Structure

|                                                                       |
| --------------------------------------------------------------------- |
| [File header](#file-header)                                           |
| [Abstract module header](#abstract-module-header) 1                   |
| ...                                                                   |
| [Abstract module header](#abstract-module-header) N                   |
| [IR module header](#ir-module-header) 1                               |
| ...                                                                   |
| [IR module header](#ir-module-header) M                               |
| [Native device code image header](#native-device-code-image-header) 1 |
| ...                                                                   |
| [Native device code image header](#native-device-code-image-header) L |
| Metadata byte table                                                   |
| Binary byte table                                                     |

#### Key Components

**Abstract Modules:** collection of device binaries that share properties,
including, but not limited to: kernel names, imported symbols, exported symbols,
aspect requirements, and specialization constants. The device binaries contained inside an abstract module must either be an IR
module or a native device code image. IR modules contain device binaries in some
known intermediate representation, such as SPIR-V, while the native device code
images can be an architecture-specific binary format. There is no requirement
that all device binaries in an abstract module are usable on the same device or
are specific to a single vendor.

**IR modules:** binary data for the corresponding module compiled to a
given IR representation, identified by the IR type field.

**Native device code images:** binary data for the corresponding
module AOT compiled for a specific device, identified by the architecture
string.

**Byte tables:** A byte table contains dynamic data, such as metadata and binary blobs. The
contents of it is generally referenced by an offset specified in the headers.

Detailed design can be found [here](https://intel.github.io/llvm/design/SYCLBINDesign.html)

## Toolchain integration

The SYCLBIN content is embedded as an image within the [offload binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h) produced by the [clang-sycl-linker](https://github.com/llvm/llvm-project/tree/main/clang/tools/clang-sycl-linker).

### Clang driver changes

- ``--sycl-link`` would trigger use of a SYCLBIN format in toolchain

### clang-sycl-linker changes

- [ ] to be updated...

The clang-linker-wrapper is responsible for doing module-splitting, metadata
extraction and linking of device binaries, as described in
[OffloadDesign.md](OffloadDesign.md). However, to support SYCLBIN files, the
clang-linker-wrapper must be able to unpack an offload binary (as described in
[ClangOffloadPackager.rst](https://github.com/intel/llvm/blob/sycl/clang/docs/ClangOffloadPackager.rst))
directly, instead of extracting it from a host binary. This should be done when
a new flag, `--syclbin`, is passed. In this case, the clang-linker-wrapper is
responsible to package the resulting device binaries and produced metadata into
the format described in [SYCLBIN binary format section](#syclbin-binary-format).

### clang-linker-wrapper changes

- [ ] to be updated...

Additionally, in this case the clang-linker-wrapper will skip the wrapping of
the device code and the host code linking stage, as there is no host code to
wrap the device code in and link.

### SYCL runtime library changes

- [ ] do we want to provide details in RFC or limit it to basic info?

Using the interfaces from the
[sycl_ext_oneapi_syclbin](../extensions/proposed/sycl_ext_oneapi_syclbin.asciidoc)
extension, the runtime must be able to parse the SYCLBIN format, as described in
the [SYCLBIN binary format section](#syclbin-binary-format). To avoid large
amounts of code duplication, the runtime uses the implementation of SYCLBIN
reading and writing implemented in LLVM.

When creating a `kernel_bundle` from a SYCLBIN file, the runtime reads the
contents of the SYCLBIN file and creates the corresponding data structure from
it.

- [ ] this part below needs to be rewritten I think....

In order for the SYCL runtime library's existing logic to use the binaries,
the runtime then creates a collection of `sycl_device_binary_struct` objects and
its constituents, pointing to the data in the parsed SYCLBIN object. Passing
these objects to the runtime library's `ProgramManager` allows it to reuse the
logic for compiling, linking and building SYCL binaries.

In the other direction, users can request the "contents" of a `kernel_bundle`.
When this is done, the runtime library must ensure that a SYCLBIN file is
available for the contents of the `kernel_bundle` and must then write the
SYCLBIN object to the corresponding binary representation in the format
described in the [SYCLBIN binary format section](#syclbin-binary-format). In cases
where the `kernel_bundle` was created with a SYCLBIN file, the SYCLBIN
representation is immediately available and can be serialized directly. In other
cases, the runtime library creates a new SYCLBIN object from the binaries
associated with the `kernel_bundle`, then serializes it and returns the result.

## Versioning and Extensibility

The SYCLBIN format is subject to change, but any such changes must come with an
increment to the version number in the header.
Additionally, any changes to the property set structure that affects the way the
runtime has to parse the contained property sets will require an increase in the
SYCLBIN version. Adding new property set names or new predefined properties only
require a SYCLBIN version change if the the SYCLBIN consumer cannot safely
ignore the property.

## Upstreaming Plan

- Phase 1: Upstream SYCLBIN format specification, including parsing/writing
- Phase 2: Add clang driver, sycl-linker and linker-wrapper support
- Phase 3: Integrate SYCLBIN support into SYCL runtime

## Opens, common todos

- [ ] Do we want to extend SYCL spec with SYCLBIN format? Do we want to somehow mention it in the RFC? Is it relevant?
- [ ] Need to add/update links in this RFC
- [ ] Polish formatting, fix typos, etc...
