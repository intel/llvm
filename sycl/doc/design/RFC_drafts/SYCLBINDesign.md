# [RFC] SYCLBIN - A format for SYCL device code

## Summary

This RFC proposes the addition of SYCLBIN, a new binary format for storing SYCL device code. The format provides a lightweight, extensible wrapper around device modules and their corresponding SYCL-specific metadata to be produced/consumed by tools/SYCL runtime.

## Purpose of this RFC

This RFC seeks community feedback on the proposed SYCLBIN binary format, including toolchain integration approach. Community input is particularly valuable regarding potential integration challenges with existing LLVM offloading implementations.

## Motivation and alternatives considered

### Requirements: SYCL-Specific metadata and modules hierarchy

The SYCL programming model requires device images to be accompanied by specific metadata necessary for SYCL runtime operation:

1. Device target triple (e.g. ``spirv64_unknown_unknown``).
2. Compiler and linker options for JIT compilation scenarios
3. List of entry points exposed by each image
4. Arrays of [property sets](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/PropertySets.md).

When a binary contains multiple images, some may share common metadata. Therefore, we require a hierarchical structure that enables metadata sharing while allowing specification of image-specific metadata.

#### Existing Formats Limitations

LLVM's offloading infrastructure supports several binary formats that can be embedded within the [Offload Binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h) format. However, these formats have various limitations that make them unsuitable for SYCL:

- **Single-module design**: Formats like Object, Bitcode, CUBIN, PTX, and SPIRV are designed for single binary or single-module IR representation, lacking hierarchical structuring capabilities for multiple images/modules.
- **Missing SYCL metadata support**: None provide native support for SYCL-specific metadata requirements.
- **Vendor constraints**: Fatbinary is NVIDIA proprietary and incompatible with SYCL's vendor-neutral approach.
- **Limited container capabilities**: [Offload Binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h) is not designed for multiple device images or hierarchical organization, with ``StringData`` insufficient for complex metadata structures (like #3 and #4 above).

### Abstraction: Simplifying Support in Offloading Tools

Another motivation for adding the SYCLBIN format is to encapsulate SYCL-specific logic within SYCL-specific toolchain components ([clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst), SYCL runtime) and isolate SYCL implementation details from general offloading tools designed to support multiple programming models.

#### Current Workflow Without SYCLBIN

Without this format, metadata transfer from compiler to runtime requires the following complicated workflow:

1. **[clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst)** uses [Offload Binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h)'s ``StringData`` (with workarounds) to store metadata items #1-#4
2. **[clang-linker-wrapper](https://github.com/llvm/llvm-project/blob/c083fa1597f1a34fcab4c2910158a288defc72f6/clang/docs/ClangLinkerWrapper.rst)** opens [Offload Binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h) files produced by [clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst) and generates [device image binary descriptors](https://github.com/llvm/llvm-project/blob/139a6bf0e448ebd7ef9bd1c26aa92018d90f8add/llvm/lib/Frontend/Offloading/OffloadWrapper.cpp#L675) in a format readable by SYCL runtime
    - Problem: This requires [clang-linker-wrapper](https://github.com/llvm/llvm-project/blob/c083fa1597f1a34fcab4c2910158a288defc72f6/clang/docs/ClangLinkerWrapper.rst) to maintain SYCL-specific format knowledge, creating unnecessary code duplication
3. **SYCL runtime** decodes metadata using this intermediate format

#### Simplified Workflow With SYCLBIN

The SYCLBIN format enables a cleaner separation of concerns:

1. **[clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst)** prepares a complete SYCLBIN containing all metadata and multiple images, embedding it as a single image within [Offload Binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h)
2. **[clang-linker-wrapper](https://github.com/llvm/llvm-project/blob/c083fa1597f1a34fcab4c2910158a288defc72f6/clang/docs/ClangLinkerWrapper.rst)** generates only host register/unregister calls without needing knowledge of SYCLBIN internals
3. **SYCL runtime** works directly with SYCLBIN format

This approach eliminates the need for [clang-linker-wrapper](https://github.com/llvm/llvm-project/blob/c083fa1597f1a34fcab4c2910158a288defc72f6/clang/docs/ClangLinkerWrapper.rst) to understand SYCL-specific formats, reducing maintenance burden and improving toolchain modularity.

### Enable modular dynamic loading of device binaries at runtime

Some applications require dynamic loading of device binaries at runtime to achieve modularity and avoid recompiling the entire application when device code changes. The SYCLBIN format provides a standardized interface between compiler-produced SYCLBIN binaries and runtime handling, enabling efficient dynamic loading scenarios.

## Design

### SYCLBIN binary format

The SYCLBIN format consists of:

- A file header with magic number and version information
- Three lists of headers: the abstract module header list, the IR module header list and native device code image header list,
containing information about the abstract modules, IR modules and native device code images respectively.
- Two byte tables containing metadata and binary data

#### File Structure

|                                   |
| --------------------------------- |
| File header                       |
| Abstract module header 1          |
| ...                               |
| Abstract module header N          |
| IR module header 1                |
| ...                               |
| IR module header M                |
| Native device code image header 1 |
| ...                               |
| Native device code image header L |
| Metadata byte table               |
| Binary byte table                 |

#### Key Components

**Abstract Modules:** collection of device binaries that share properties, including, but not limited to: exported symbols, [aspect requirements](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/OptionalDeviceFeatures.md), and [specialization constants](https://github.khronos.org/SYCL_Reference/iface/specialization-constants.html). The device binaries contained inside an abstract module must either be an IR module or a native device code image. IR modules contain device binaries in some known intermediate representation, such as SPIR-V, while the native device code images can be an architecture-specific binary format. There is no requirement that all device binaries in an abstract module are usable on the same device or are specific to a single vendor.

**IR modules:** binary data for the corresponding module compiled to a given IR representation, identified by the IR type field.

**Native device code images:** binary data for the corresponding module AOT compiled for a specific device, identified by the architecture
string.

**Byte tables:** A byte table contains dynamic data, such as metadata and binary blobs. The contents of it is generally referenced by an offset specified in the headers.

Detailed design can be found [here](https://intel.github.io/llvm/design/SYCLBINDesign.html)

## Toolchain integration

The SYCLBIN content can either be embedded as an image within the [Offload Binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h) produced by the [clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst) or outputted directly as standalone SYCLBIN files.

This integration approach allows SYCLBIN to leverage the existing [Offload Binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h) infrastructure while maintaining its specialized format for SYCL-specific requirements.

### [clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst) changes

The [clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst) is responsible for module-splitting, metadata extraction (symbol tables, property sets, etc.) and linking of device binaries. To support SYCLBIN, it must be able to:

- Pack device binaries and extracted metadata into the SYCLBIN format
- Embed the resulting SYCLBIN into an OffloadBinary container or output standalone SYCLBIN files
- Support linking multiple SYCLBIN binaries together

### [clang-linker-wrapper](https://github.com/llvm/llvm-project/blob/c083fa1597f1a34fcab4c2910158a288defc72f6/clang/docs/ClangLinkerWrapper.rst) changes

The [clang-linker-wrapper](https://github.com/llvm/llvm-project/blob/c083fa1597f1a34fcab4c2910158a288defc72f6/clang/docs/ClangLinkerWrapper.rst) shall support two operational modes:

- **Standalone SYCLBIN output:** Output SYCLBIN binaries directly, skipping device code wrapping and host code linking stages.
- **Host linking:** Generate host register/unregister calls for SYCL runtime access to SYCLBIN binaries and link with host code.

### SYCL runtime library changes

The runtime must be able to parse the SYCLBIN format, using the implementation of SYCLBIN
reading and writing functionality.

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
- Phase 2: Add clang driver, [clang-sycl-linker](https://github.com/llvm/llvm-project/blob/main/clang/docs/ClangSYCLLinker.rst) and [clang-linker-wrapper](https://github.com/llvm/llvm-project/blob/c083fa1597f1a34fcab4c2910158a288defc72f6/clang/docs/ClangLinkerWrapper.rst) support
- Phase 3: Integrate SYCLBIN support into SYCL runtime
