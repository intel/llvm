# SYCLBIN - A format for separately compiled SYCL device code

Some applications may want to dynamically load device binaries at runtime, e.g.
for modularity and to avoid having to recompile the entire application. However,
doing so through the
[sycl_ext_oneapi_kernel_compiler](https://github.com/intel/llvm/blob/sycl/doc/extensions/experimental/sycl_ext_oneapi_kernel_compiler.asciidoc)
extension can be unnecessarily expensive, if the systems utilizing this
modularity are able to compile the binaries separate from the application's
execution.

To facilitate that a new SYCLBIN format is needed to define the interface between
the compiler-produced binaries and the runtime's handling of it. This necessity
comes from the overall design of our SYCL toolchain where runtime relies on
compiler-provided information/metadata to implement various features (like
support for specialization constants or shared libraries), i.e. device code alone
is not enough.

This design document details the SYCLBIN binary format used for storing SYCL
device binaries to be loaded dynamically by the SYCL runtime. It also details
how the toolchain produces, links and packages these binaries, as well as how
the SYCL runtime library handles files of this format.


## SYCLBIN binary format

The files produced by the new compilation path will follow the format described
in this section. The intention of defining a new format for these is to give
the DPC++ implementation an extendable and lightweight wrapper around the
multiple modules and corresponding metadata captured in the SYCLBIN file.
The content of the SYCLBIN may be contained as an entry in the offloading binary
format produced by the clang-offload-packager, as described in
[ClangOffloadPackager.rst](https://github.com/intel/llvm/blob/sycl/clang/docs/ClangOffloadPackager.rst).

The format of these files consist of a [file header](#file-header), containing
general information about the file, followed by three lists of headers: The
[abstract module header](#abstract-module-header) list, the
[IR module header](#ir-module-header) list and
[native device code image header](#native-device-code-image-header) list,
containing information about the [abstract modules](#abstract-module),
[IR modules](#ir-module) and
[native device code images](#native-device-code-image) respectively.

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

### File header

The file header segment appears as the first part of the SYCLBIN file. Like many
other file-formats, it defines a magic number to help identify the format, which
is 0x53594249 (or "SYBI".) Immediately following the magic number is the version
number, which is used by SYCLBIN consumers when parsing data in the rest of the
file.

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


#### Global metadata

The global metadata entry contains a single property set with the identifying
name "SYCLBIN/global metadata", as described in the
[PropertySets.md](PropertySets.md#syclbinglobal-metadata) design document.


### Abstract module

An abstract module is a collection of device binaries that share properties,
including, but not limited to: kernel names, imported symbols, exported symbols,
aspect requirements, and specialization constants.

The device binaries contained inside an abstract module must either be an IR
module or a native device code image. IR modules contain device binaries in some
known intermediate representation, such as SPIR-V, while the native device code
images can be an architecture-specific binary format. There is no requirement
that all device binaries in an abstract module are usable on the same device or
are specific to a single vendor.


#### Abstract module header

A abstract module header contains the following fields in the stated order:

| Type       | Description                                                                                |
| ---------- | ------------------------------------------------------------------------------------------ |
| `uint64_t` | Byte offset of the metadata from the start of the metadata byte table.                     |
| `uint64_t` | Byte size of the metadata in the metadata byte table.                                      |
| `uint32_t` | Number of IR modules.                                                                      |
| `uint32_t` | Index of the first IR module header in the IR module header array.                         |
| `uint32_t` | Number of native device code images.                                                       |
| `uint32_t` | Index of the first native device code images header native device code image header array. |

#### Abstract module metadata

An abstract module metadata entry contains any number of property sets, as
described in [PropertySets.md](PropertySets.md), excluding:

* ["SYCLBIN/global metadata"](PropertySets.md#syclbinglobal-metadata)
* ["SYCLBIN/ir module metadata"](PropertySets.md#syclbinir-module-metadata)
* ["SYCLBIN/native device code image module metadata"](PropertySets.md#syclbinnative-device-code-image-metadata)


#### IR module

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


#### Native device code image

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


### Byte tables

A byte table contains dynamic data, such as metadata and binary blobs. The
contents of it is generally referenced by an offset specified in the headers.


### SYCLBIN version changelog

The SYCLBIN format is subject to change, but any such changes must come with an
increment to the version number in the header and a subsection to this section
describing the change.

Additionally, any changes to the property set structure that affects the way the
runtime has to parse the contained property sets will require an increase in the
SYCLBIN version. Adding new property set names or new predefined properties only
require a SYCLBIN version change if the the SYCLBIN consumer cannot safely
ignore the property.


#### Version 1

 * Initial version of the layout.


## Clang driver changes

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
`-fsyclbin` will cause `-fsycl-device-only` to be considered unused.

The behavior is dependent on using the clang-linker-wrapper. As the current
default offload compilation behavior is using the old offload model (driver
based), this option currently requires `--offload-new-driver` to be set.
</td>
</tr>
<tr>
<td>`--offload-rdc`</td>
<td>This is an alias of `-fgpu-rdc`.</td>
</tr>
</table>

Additionally, `-fsycl-link` should work with .syclbin files. Semantics of how
SYCLBIN files are linked together is yet to be specified.


## clang-linker-wrapper changes

The clang-linker-wrapper is responsible for doing module-splitting, metadata
extraction and linking of device binaries, as described in
[OffloadDesign.md](OffloadDesign.md). However, to support SYCLBIN files, the
clang-linker-wrapper must be able to unpack an offload binary (as described in
[ClangOffloadPackager.rst](https://github.com/intel/llvm/blob/sycl/clang/docs/ClangOffloadPackager.rst))
directly, instead of extracting it from a host binary. This should be done when
a new flag, `--syclbin`, is passed. In this case, the clang-linker-wrapper is
responsible to package the resulting device binaries and produced metadata into
the format described in [SYCLBIN binary format section](#syclbin-binary-format).
Additionally, in this case the clang-linker-wrapper will skip the wrapping of
the device code and the host code linking stage, as there is no host code to
wrap the device code in and link.

*TODO:* Describe the details of linking SYCLBIN files.


## SYCL runtime library changes

Using the interfaces from the
[sycl_ext_oneapi_syclbin](../extensions/proposed/sycl_ext_oneapi_syclbin.asciidoc)
extension, the runtime must be able to parse the SYCLBIN format, as described in
the [SYCLBIN binary format section](#syclbin-binary-format). To avoid large
amounts of code duplication, the runtime uses the implementation of SYCLBIN
reading and writing implemented in LLVM.

When creating a `kernel_bundle` from a SYCLBIN file, the runtime reads the
contents of the SYCLBIN file and creates the corresponding data structure from
it. In order for the SYCL runtime library's existing logic to use the binaries,
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

