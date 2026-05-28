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

A SYCLBIN file on disk is an
[OffloadBinary](https://github.com/intel/llvm/blob/sycl/clang/docs/OffloadingDesign.rst#creating-fat-objects)
whose every Entry has `ImageKind == IMG_SYCLBIN` and
`OffloadKind == OFK_SYCL`. The OffloadBinary already supplies the outer
magic, length, alignment and string-data plumbing; SYCLBIN reuses this
envelope rather than defining a parallel container. The on-disk header
(`Header`/`Entry`/`StringEntry`) and `0x10FF10AD` magic bytes are defined in
[`llvm/include/llvm/Object/OffloadBinary.h`](https://github.com/intel/llvm/blob/sycl/llvm/include/llvm/Object/OffloadBinary.h).

> **Note**: this section describes the **v2** layout, written by current
> toolchains. The earlier **v1** layout (with a SYBI-magic inner header
> and SYCLBIN-private header tables) is no longer produced but is still
> accepted by the runtime and tools for backward compatibility — see
> [SYCLBIN version changelog](#syclbin-version-changelog).

In v2 the abstract-module structure is encoded entirely through OffloadBinary
StringData keys on each Entry. Every Entry plays one of four roles:

| `role` value      | Image bytes                                                                                                  | StringData keys                              |
| ----------------- | ------------------------------------------------------------------------------------------------------------ | -------------------------------------------- |
| `global_metadata` | Serialized `PropertySetRegistry` carrying the [global metadata](#global-metadata).                            | `role`                                       |
| `am_metadata`     | Serialized `PropertySetRegistry` carrying the [abstract module metadata](#abstract-module-metadata).          | `role`, `am_index`                           |
| `ir`              | `[u64 LE metadata_size][PropertySetRegistry of metadata_size bytes][raw IR bytes]`                            | `role`, `am_index`, `ir_type`, `triple`      |
| `native`          | `[u64 LE metadata_size][PropertySetRegistry of metadata_size bytes][raw native bytes]`                        | `role`, `am_index`, `arch`, `triple`         |

Exactly one `global_metadata` Entry is required. Each abstract module is
identified by a non-negative integer `am_index`; the file must contain
exactly one `am_metadata` Entry per `am_index`, plus zero or more `ir` /
`native` Entries pointing at that index. The set of used `am_index`
values must be contiguous starting at 0.

|                                                            |
| ---------------------------------------------------------- |
| OffloadBinary header                                       |
| Entry 1 (`role=global_metadata`)                           |
| Entry 2 (`role=am_metadata`, `am_index=0`)                 |
| Entry k (`role=ir`/`native`, `am_index=0`, ...)            |
| ...                                                        |
| Entry m (`role=am_metadata`, `am_index=N-1`)               |
| Entry n (`role=ir`/`native`, `am_index=N-1`, ...)          |

The OffloadBinary itself (header, entry table, string table) is described in
[OffloadingDesign.rst -> Creating Fat Objects](https://github.com/intel/llvm/blob/sycl/clang/docs/OffloadingDesign.rst#creating-fat-objects)
and declared in
[`llvm/include/llvm/Object/OffloadBinary.h`](https://github.com/intel/llvm/blob/sycl/llvm/include/llvm/Object/OffloadBinary.h);
SYCLBIN inherits its 8-byte alignment, magic number and little-endian field
layout from there.

The order of `am_metadata`, `ir` and `native` Entries is not significant — the
reader reconstructs the abstract-module structure from `am_index`. Tools that
emit SYCLBIN should keep entries belonging to the same abstract module
contiguous for human readability of `syclbin-dump` output.

### Global metadata

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

Each abstract module is represented in the file by one Entry with
`role=am_metadata` plus zero or more `ir` / `native` Entries that share the
same `am_index`.

#### Abstract module metadata

An abstract module metadata entry contains any number of property sets, as
described in [PropertySets.md](PropertySets.md), excluding:

* ["SYCLBIN/global metadata"](PropertySets.md#syclbinglobal-metadata)
* ["SYCLBIN/ir module metadata"](PropertySets.md#syclbinir-module-metadata)
* ["SYCLBIN/native device code image module metadata"](PropertySets.md#syclbinnative-device-code-image-metadata)


#### IR module

An IR module contains the binary data for the corresponding module compiled to a
given IR representation, identified by the IR type field.

An IR module is encoded as one Entry with `role=ir` whose StringData carries
`am_index` (the owning abstract module), `ir_type` (decimal IR type tag) and
`triple` (LLVM target triple string). Its image bytes are
`[u64 LE metadata_size][serialized PropertySetRegistry of metadata_size bytes][raw IR bytes]`.

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

A native device code image is encoded as one Entry with `role=native` whose
StringData carries `am_index` (the owning abstract module), `arch`
(architecture string) and `triple` (LLVM target triple string). Its image
bytes are
`[u64 LE metadata_size][serialized PropertySetRegistry of metadata_size bytes][raw native device code image bytes]`.

##### Native device code image metadata

A native device code image metadata entry contains a single property set with
the identifying name "SYCLBIN/native device code image module metadata", as
described in the
[PropertySets.md](PropertySets.md#syclbinnative-device-code-image-metadata)
design document.


### SYCLBIN version changelog

The SYCLBIN format is subject to change, but any such changes must come with an
increment to the version number and a subsection to this section describing
the change.

Additionally, any changes to the property set structure that affects the way the
runtime has to parse the contained property sets will require an increase in the
SYCLBIN version. Adding new property set names or new predefined properties only
require a SYCLBIN version change if the the SYCLBIN consumer cannot safely
ignore the property.


#### Version 1 (legacy, read-only)

 * Initial layout: an OffloadBinary envelope wrapping a single Entry whose
   image bytes start with the `0x53594249` ("SYBI") magic and contain
   SYCLBIN-private FileHeader / AbstractModuleHeader / IRModuleHeader /
   NativeDeviceCodeImageHeader tables, a metadata byte table and a binary
   byte table. Current toolchains no longer produce this layout, but
   `llvm::object::SYCLBIN::read` continues to accept it so that previously
   built `.syclbin` artifacts remain loadable.


#### Version 2

 * Replaced the SYBI-magic inner header tables with direct use of multi-entry
   OffloadBinary. The on-disk file is now a single OffloadBinary; abstract
   modules, IR modules and native device code images are each represented by
   their own Entry, and abstract-module grouping is encoded via the
   `am_index` StringData key on each Entry. The discriminator between v1 and
   v2 at read time is whether the first Entry's image bytes start with the
   legacy SYBI magic.


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
llvm-offload-binary invocation to clang-linker-wrapper together with the new
`--syclbin` flag.

Setting this option will override `-fsycl`. Passing`-fsycl-device-only` with
`-fsyclbin` will cause `-fsyclbin` to be considered unused.

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
[OffloadingDesign.rst -> Creating Fat Objects](https://github.com/intel/llvm/blob/sycl/clang/docs/OffloadingDesign.rst#creating-fat-objects))
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

