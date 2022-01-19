# Huge object files support implementation design

This document describes the implementation design for huge object files 

## Problem statement

There are applications, which are huge enough to under some circumstances reach
limitations of executable sizes. On Linux, the issue manifests as
`relocation truncated to fit: R_X86_64_PC32` error emitted by linker for
certain symbols and sections. On Windows, the issue manifests as
`is not a valid Win32 application.` error emitted during application startup.

The limits are: for Linux: object file with any section being bigger than 2GB;
for Windows: final executable being bigger than 2GB.

There are several things which can lead to this issue:
- application is huge enough by itself;
- application is compiled with `-g` flag, i.e. debug info is enabled;
- application is compiled for several targets
  (i.e. `-fsycl-targerts=spir64,spir64_gen -fsycl-target-backend=spir64_gen "-device *"`)
- application has a lot of function which are used in different kernels, which
  were split into separate device images due to device code split enabled
- application contains a lot of `parallel_for(range, ...)` calls, which are
  subject of range rounding feature and they are being duplicated by the
  implementation
- combination of reasons above

## Possible solutions

On Linux, it is possible so simply apply proper linker script to put sections
in a right order to avoid the issue.

Apparently, that won't work on Windows and moreover, [PE Format][1] has a hard
limit on executables sizes which is 2GB:

> This document specifies the structure of executable (image) files and object
> files under the Microsoft Windows family of operating systems.
>
> The optional header magic number determines whether an image is a PE32 or
> PE32+ executable.
> PE32+ images allow for a 64-bit address space while limiting the image size to
> 2 gigabytes. 

[1]: https://docs.microsoft.com/en-us/windows/win32/debug/pe-format

Therefore, on Windows the only way to solve the issue is to reduce size of the
resulting binary, which can be done by outlining device code into a separate
file or several files.

## Linker script solution

The issue with relative relocations not being able to "reach" particular
symbol happens when we have a huge device code section which appears to be in
between of two other sections referecing each other with 32-bit relative
relocations. An example:

```
# sections:

.text
.__CLANG_OFFLOAD_BUNDLE__spir64
.rodata

# relocations:

RELOCATION RECORDS FOR [.text]:
OFFSET           TYPE              VALUE
0000000000000002 R_X86_64_PC32     .rodata+0x00000000000002c0
```

The reason for such sections placement is the algorithm of how linker works:

> If you do not use a SECTIONS command in your linker script, the linker will
> place each input section into an identically named output section in the order
> that the sections are first encountered in the input files. If all input
> sections are present in the first file, for example, the order of sections in
> the output file will match the order in the first input file. The first
> section will be at address zero.
>
> [3.6 SECTIONS command][2]

[2]: https://sourceware.org/binutils/docs/ld/SECTIONS.html

Therefore, the solution is to adjust the default linker script to explicitly
put `__CLANG_OFFLOAD_BUNDLE__*` sections at the end of the binary so they are
not inserted between any other two sections.

## Generic solution

This solution will work on both Linux and Windows, but it requires more
involvment from user. The idea is to outline device code into a separate file,
so the size of the final binary is not exceeding limitations.

In turn, such outlining can also be done in different ways:

1. device code is outlined into a separate shared library, which is linked to
   the main application.

   Pros:
     - no need to develop custom mechanisms for dependencies tracking, library
       searching and error reporting;
   Cons:
     - shared libraries have the same 2GB limit on Windows, which means that
       further splitting of such library could be required;

2. device code is outlined into a separate file with custom format.

   Pros:
     - no limits on that file size, no further splitting would be required;
   Cons:
     - more mechanism will have to be developed: searhing for that file,
       efficient partial loading of it, error reporting in case it is not found,
       etc.

In both cases, user will have to invoke the compiler/linker with additional
command line arguments to instruct the toolchain to do the split. Such split
also complicates packaging/installing/deployment of the application due to extra
files being produced by the toolchain.

Note: we specifically do not consider a case when compiler implicitly generates
several files as a result of compilation, because it won't be possible properly
integrate that into existing build system.

The following sections will describe in more details how each of the approaches
can be implemented. The first approach requires less changes in the toolchain
than the second, but at the same time it can be less user-friendly in cases
where device code alone doesn't fit into 2GB and has to be split further.

### Outlining device code into a shared library

Two new command line options are introduced: `-fsycl-only-link-host` and
`-fsycl-only-link-device`. Each option is only effective on link step and it
instructs the compiler to only link either host or device part of input fat
binaries.

**TODO**. Can we somehow re-use existing `-fsycl-device-only` and accompany it
with something like `-fsycl-host-only`?

Therefore, it would look like this for user:
```
clang++ -fsycl app.cpp -c -o a.o
clang++ -fsycl a.o -fsycl-only-link-device -o a.device.so
clang++ -fsycl a.o -fsycl-only-link-host -la.device -o app.out
```

User is responsible for linking shared library with device code to an
application or otherwise no kernels will be available at runtime.

Using existing shared libraries mechanism allows us to re-use:
- OS-provided mechanism for searching for shared library through well-known
  algorithms (`LD_LIBRARY_PATH`/`PATH`)
- OS-provided tools for displaying dependencies (like `ldd`)
- OS-provided automatic loading of a library with device code
  **TODO**. Need to check if Windows still loads a library and calls
  constructors out of it even if the app has no symbol dependencies on that
  library
- OS-provided mechanisms for error reporting in case required library is missing

Further splitting of a device code can be performed by user in different ways:
- invoking `clang++ -fsycl -fsycl-only-link-device` with different values of
  `-fsycl-targets` flag to create different device code libraries for different
  targerts;
- explicitly creating several device libraries instead of just one from a list
  of source files;

#### Changes to Driver

We need to implement two new command line options, which are only effective
during link stage:

- `-fsycl-link-only-host`. Instructs the compiler to extract and link only host
  part of input fat binaries.
- `-fsycl-link-only-device`. Instructs the compiler to extract and link only
  device part of input fat binaries. If `-fsycl-targets` is also present, the
  compiler will only link sections of input fat binaries, which correspond to
  specified targets, allowing user to create several libraries from the same set
  of fat object where each library is built for different set of targets.

### Outlining device code into a separate file

In this mode, user will still have to invoke the compiler for linking two times,
i.e. separately for host and device parts of an application. However, the main
difference from the variant with shared libraries is that all device code is
stored in a single file with custom format, which don't have any size
limitations. That means that users won't have to further split device code if it
alone exceeds 2GB.

The downside of this approach lies on implementation side: we have to
re-implement a lot of existing things ourselves, which includes:
- embedding dependencies information about the file with device code into an
  application which is being produced by the compiler: runtime needs to somehow
  know which file it should open to get device code;
- loading device code file at runtime and error reporting in case it can't be
  found;
- mechanisms for injecting different device code file at runtime for debugging
  purposes, i.e. alternatives for `LD_LIBRARY_PATH`,`PATH`,`LD_PRELOAD`, etc.
- tools for displaying dependencies of a SYCL app on device code: which file
  is needed for the app, where the app would look for it, etc., i.e. `ldd`
  equvalent;
- tools for exploring device code file for debugging purposes, equivalents of
  `readelf`, `objdump`, `dumpbin`, etc.

On top of that, format of the device code file should allow partial and/or lazy
reading of it, i.e. if we have device code for several targets encoded there, we
shouldn't have to read the whole file to extract device code for a particular
target.

Some of those questions can be resolved by re-using existing techologies like
using ELF format for device code file, which will automatically give us partial/
lazy loading and support for tools for exploring of device code. However, this
doesn't significantly reduce complexity comparing to shared libraries for device
code approach.

#### Changes to Driver

As with the previous variant, we need to implement the same two command line
options, but with slightly different meaning

- `-fsycl-link-only-host`. Instructs the compiler to extract and link only host
  part of input fat binaries.
- `-fsycl-link-only-device`. Instructs the compiler to extract and link only
  device part of input fat binaries.

If we decide to support both ways of device code storage (shared libraries and
custom file format), then we can distinguish between them by either parsing an
extension of a specified output file or by providing a one more command line
option to choose between one approach or another.

































mjk

