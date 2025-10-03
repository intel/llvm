# [RFC] SYCLBIN - A format for SYCL device code

## Summary

This RFC proposes the addition of SYCLBIN, a new binary format for storing SYCL device code. The format provides a lightweight, extensible wrapper around device modules and their corresponding SYCL-specific metadata to be produced/consumed by tools/SYCL runtime.

## Purpose of this RFC

This RFC seeks community feedback on the proposed SYCLBIN binary format design, including:

- The binary format specification and layout
- Toolchain integration approach and new compiler flags

Community input is particularly valuable regarding potential integration challenges with existing LLVM offloading implementations.

## Motivation and alternatives considered

### Metadata unique for SYCL programming model

- [ ] This section needs re-writing...

LLVM offloading infrastructure supports the following binary formats: Object, Bitcode, Cubin, Fatbinary, PTX and SPIRV which could be placed into OffloadBinary format. None of it satisfies the needs of SYCL programming model.

- [ ] Steffen, I need to discuss with you, why other existing formats did not satisfy our needs. I think we need to provide short summary why each format doesn't work for us somewhere in this section.

Specifically, SYCL needs to keep the following metadata necessary for SYCL runtime, which is not supported by any of existing formats:

1. Device target triple (e.g. spirv64_unknown_unknown).
2. Compiler and linker options to pass to JIT compiler in case of JITing.
3. List of entry points exposed by an image
4. Arrays of property sets.

While #1 and #2 can be saved to StringData of OffloadBinary, #3 requires additional handling, since StringData serialization infrastructure assumes that value is a single null-terminated string, so to restore multiple null-terminated strings from StringData format, they need to be concatenated with split symbol and then split during deserialization.

[Property sets](https://github.com/intel/llvm/blob/sycl/sycl/doc/design/PropertySets.md) (#4) would be even more complicated.

### Abstraction: simplify support in offloading tools

Another motivation to add SYCLBIN format is to encapsulate SYCL-specific logic to SYCL-specific parts of toolchain (clang-sycl-linker, SYCL runtime) and hide SYCL specifics from offloading tools intended to support multiple programming models. Without this format, we would need to use the following workflow to pass metadata (#1 - #4) from compiler to runtime:

1. clang-sycl-linker would use OffloadingImage’s StringData to save metadata #1-#4.
Problem: OffloadingImage’s StringData is not intended for composite objects like arrays or property sets.
2. clang-linker-wrapper would open OffloadingImages prepared by clang-sycl-linker and generate device image binary descriptor for each image in some format that SYCL runtime could read.
Problem: clang-linker-wrapper needs to maintain SYCL-specific formats necessary for SYCL runtime, which means unnecessary duplication.
3. Then SYCL runtime would use this format to decode metadata.

If SYCLBIN is accepted, then the scheme could be simplified, resolving problems highlighted above:

1. clang-sycl-linker would prepare SYCLBIN with all metadata encoded put it inside OffloadingImage as image.
2. clang-linker-wrapper would generate only host register and unregister calls, but would know nothing about what’s inside SYCLBIN.
3. SYCL runtime would work with SYCLBIN directly.

### Enable modular dynamic loading of device binaries at runtime

Some applications may want to dynamically load device binaries at runtime, e.g. for modularity and to avoid having to recompile the entire application. To facilitate that SYCLBIN format defines the interface between
the compiler-produced binaries and the runtime's handling of it.

## Detailed Design

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

The headers and each byte table are all aligned to 8 bytes. The fields in the
headers use C/C++ type notation, including the fixed-size integer types defined
in the `<cstdint>` header, and will have the same size and alignment. For
consistency, all these types use little endian layout.

#### Component Details

- [ ] Do we want to provide that level of details in the RFC, or it is better to clean it up to
keep only key info and for details provide reference to design document?

##### File header

| Type       | Description                                                                   |
| ---------- | ----------------------------------------------------------------------------- |
| `uint32_t` | Magic number. (0x53594249)                                                    |
| `uint32_t` | SYCLBIN version number.                                                       |
| `uint32_t` | Number of abstract modules.                                                   |
| `uint32_t` | Number of IR modules.                                                         |
| `uint32_t` | Number of native device code images.                                          |
| `uint64_t` | Byte size of the metadata byte table.                                         |
| `uint64_t` | Byte size of the binary byte table.                                           |
| `uint64_t` | Byte offset of the global metadata from the start of the metadata byte table. |
| `uint64_t` | Byte size of the global metadata.                                             |

##### Global metadata

The global metadata entry contains a single property set with the identifying
name "SYCLBIN/global metadata", as described in the
[PropertySets.md](PropertySets.md#syclbinglobal-metadata) design document.

##### Abstract module

An abstract module is a collection of device binaries that share properties,
including, but not limited to: kernel names, imported symbols, exported symbols,
aspect requirements, and specialization constants.

The device binaries contained inside an abstract module must either be an IR
module or a native device code image. IR modules contain device binaries in some
known intermediate representation, such as SPIR-V, while the native device code
images can be an architecture-specific binary format. There is no requirement
that all device binaries in an abstract module are usable on the same device or
are specific to a single vendor.

##### Abstract module header

A abstract module header contains the following fields in the stated order:

| Type       | Description                                                                                |
| ---------- | ------------------------------------------------------------------------------------------ |
| `uint64_t` | Byte offset of the metadata from the start of the metadata byte table.                     |
| `uint64_t` | Byte size of the metadata in the metadata byte table.                                      |
| `uint32_t` | Number of IR modules.                                                                      |
| `uint32_t` | Index of the first IR module header in the IR module header array.                         |
| `uint32_t` | Number of native device code images.                                                       |
| `uint32_t` | Index of the first native device code images header native device code image header array. |

##### Abstract module metadata

An abstract module metadata entry contains any number of property sets, as
described in [PropertySets.md](PropertySets.md), excluding:

- ["SYCLBIN/global metadata"](PropertySets.md#syclbinglobal-metadata)
- ["SYCLBIN/ir module metadata"](PropertySets.md#syclbinir-module-metadata)
- ["SYCLBIN/native device code image module metadata"](PropertySets.md#syclbinnative-device-code-image-metadata)

##### IR module

An IR module contains the binary data for the corresponding module compiled to a
given IR representation, identified by the IR type field.

##### IR module header

A IR module header contains the following fields in the stated order:

| Type       | Description                                                              |
| ---------- | ------------------------------------------------------------------------ |
| `uint64_t` | Byte offset of the metadata from the start of the metadata byte table.   |
| `uint64_t` | Byte size of the metadata in the metadata byte table.                    |
| `uint64_t` | Byte offset of the raw IR bytes from the start of the binary byte table. |
| `uint64_t` | Byte size of the raw IR bytes in the binary byte table.                  |

##### IR module metadata

An IR module metadata entry contains a single property set with the identifying
name "SYCLBIN/ir module metadata", as described in the
[PropertySets.md](PropertySets.md#syclbinir-module-metadata) design document.

##### Native device code image

An native device code image contains the binary data for the corresponding
module AOT compiled for a specific device, identified by the architecture
string.  The runtime library will attempt to map these to the architecture
enumerators in the
[sycl_ext_oneapi_device_architecture](../extensions/experimental/sycl_ext_oneapi_device_architecture.asciidoc)
extension.

##### Native device code image header

A native device code image header contains the following fields in the stated
order:

| Type       | Description                                                                         |
| ---------- | ----------------------------------------------------------------------------------- |
| `uint64_t` | Byte offset of the metadata from the start of the metadata byte table.              |
| `uint64_t` | Byte size of the metadata in the metadata byte table.                               |
| `uint64_t` | Byte offset of the device code image bytes from the start of the binary byte table. |
| `uint64_t` | Byte size of the device code image bytes in the binary byte table.                  |

##### Native device code image metadata

A native device code image metadata entry contains a single property set with
the identifying name "SYCLBIN/native device code image module metadata", as
described in the
[PropertySets.md](PropertySets.md#syclbinnative-device-code-image-metadata)
design document.

##### Byte tables

A byte table contains dynamic data, such as metadata and binary blobs. The
contents of it is generally referenced by an offset specified in the headers.

## Toolchain integration

The content of the SYCLBIN may be contained as an image in the [offload binary](https://github.com/llvm/llvm-project/blame/main/llvm/include/llvm/Object/OffloadBinary.h) produced by the [clang-sycl-linker](https://github.com/llvm/llvm-project/tree/main/clang/tools/clang-sycl-linker).

### Clang driver changes

- [ ] This needs to be rewritten...

The clang driver needs to accept the following new flags:

<table>
<tr>
<th>Option</th>
<th>Description</th>
</tr>
<tr>
<td>`-fsyclbin`</td>
<td>
If this option is set, the output of the invocation is a SYCLBIN file with the
.syclbin file extension. This skips the host-compilation invocation of the
typical `-fsycl` pipeline, instead passing the output of the
clang-offload-packager invocation to clang-linker-wrapper together with the new
`--syclbin` flag.

Setting this option will override `-fsycl`. Passing`-fsycl-device-only` with
`-fsyclbin` will cause `-fsyclbin` to be considered unused.

The behavior is dependent on using the clang-linker-wrapper.
</td>
</tr>
<tr>
<td>`--offload-rdc`</td>
<td>This is an alias of `-fgpu-rdc`.</td>
</tr>
</table>

Additionally, `-fsycl-link` should work with .syclbin files. Semantics of how
SYCLBIN files are linked together is yet to be specified.

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
