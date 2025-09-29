# Debugging DPC++

## Building DPC++ in debug mode

To build DPC++ in debug mode you can simply pass `-t Debug` to `configure.py`,
however debug builds can be quite large and slow so using the following `CMake`
options may help:

- `-DLLVM_USE_SPLIT_DWARF=ON`: Use `-gsplit-dwarf`, this splits some of the
  debug information out of the object files into their own separate files,
  which reduces the size of the object files the linker has to load (see
  [DebugFission](https://gcc.gnu.org/wiki/DebugFission)).
- `-DLLVM_PARALLEL_LINK_JOBS=4`: Reduce the number of link jobs running in
  parallel, to avoid running out of RAM when linking large debug build objects.
- `-DLLVM_USE_LINKER=lld`: Use the LLVM linker `lld` instead of the GNU linker
  `ld` as `lld` is usually faster.
- `-DLIBCLC_CUSTOM_LLVM_TOOLS_BINARY_DIR=<path/to/release/build/bin>`: Use a
  separate release build of DPC++ as the compiler for building the bitcode
  libraries (`libclc` and `libdevice`), it normally uses the compiler from the
  same build, but debug clang is quite slow so using a separate release clang
  can be worth it.
- `-DLLVM_OPTIMIZED_TABLEGEN=ON`: Build the tablegen tools separately in release
  mode. Debug tablegen tools are quite slow so using release mode versions can
  significantly speed up the build.

For more information on any of the `LLVM_` variables, or possibly other helpful
options, refer to the [LLVM CMake
documentation](https://llvm.org/docs/CMake.html#llvm-related-variables).

## Tracing the SYCL runtime

The SYCL runtime is built on top of the Unified Runtime API, using the following
environment variable prints out all the calls to the Unified Runtime emitted by
the SYCL runtime, which can help understand runtime behavior:

- `SYCL_UR_TRACE=`: Enables SYCL runtime tracing
  - `1` for basic tracing, `2` for UR call tracing, `-1` for everything.

## Debugging the compiler

### Clang Driver

- `-###`: Prints each command emitted by the clang driver during compilation
  without running them.
  - Can be used to manually replay a compilation command step-by-step to narrow
    down where a crash happened.

### Middle-end and back-end

LLVM has a number of ways to debug LLVM IR passes and lower, the following
options illustrate a few of them:

- `-save-temps`: Dump all compilation intermediary files. Adding `-v` will also
  print the sub-commands generating the intermediary files.
- `-mllvm -print-after-all`: Dump modules before and after each pass of the compilation pipeline
  - Often produces a huge amount of data but can be helpful to track down where
    something is introduced in the IR or assembly.
  - To reduce the output size `-mllvm -filter-print-funcs=<function name>`
    can be used to filter which functions in the module are printed.
- `-mllvm -opt-bisect-limit=<number>`: Find which optimization pass is causing issues.
  - In cases where a bug only occurs at a certain optimization level this flag
    can help bisect which optimization pass is introducing the issue. Using
    it will print every optimization pass being run with a number, passing that
    number to the flag makes the optimization pipeline stop any pass after that
    number. This allows for manual bisection of the issue by adjusting the
    number passed to the flag. Note that in-lining may interfere with this
    because it changes the number of `FunctionPass` being run.
- `-mllvm -debug-only=<tag>`: Enable debug output for given LLVM pass or components.
  - In LLVM it can be defined as follows `#define DEBUG_TYPE "regalloc"`, this
    hooks into `-debug-only` allowing you to enable debug output for the pass
    defined in that file and any other that uses the same string as
    `DEBUG_TYPE`. For example `-mllvm -debug-only=regalloc` will enable debug
    output for all the register allocation passes. For more details on this
    refer to the [LLVM
    documentation](https://www.llvm.org/docs/ProgrammersManual.html#the-llvm-debug-macro-and-debug-option).

## Extracting device code

This section is focused on AOT support for Nvidia and AMD, but some of the tips
can also be used for other targets.

### Compiling directly to device assembly

Using the flags `-fsycl-device-only -fsycl-device-obj=asm -S` you can instruct
the SYCL compiler to output the assembly for the device code. For example when
targeting Nvidia, the following command will output the device code PTX:

```
clang++ -fsycl -fsycl-targets=nvidia_gpu_sm_61 -fsycl-device-only -fsycl-device-obj=asm -S a.cpp -o a.ptx
```

### Extract device binary from SYCL application

#### Using `SYCL_DUMP_IMAGES`

The first way of extracting the device code is to run the application with the
environment `SYCL_DUMP_IMAGES` set to `1`.

This will dump all the images available in the SYCL binary with names such as
`sycl_amdgcn1.bin` and `sycl_nvptx641.bin`.

These are target specific fat binaries from which the device code can then be
extracted.

#### Using `clang-offload-extract`


Instead of using the environment variable to dump the images, it is also
possible to extract them manually with `clang-offload-extract`, assuming we have
a binary named `main`:

```
# Extract bundled device binary from host ELF file
$ ./bin/clang-offload-extract main --output=extracted_main
```

This will output files such as `extracted_main.0`, these extracted files are
target specific fat binaries from which the device code can then be extracted.

### Extracting device code from target fat binaries

#### CUDA fat binaries

CUDA fat binaries can be analyzed with the
[`cuobjdump`](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html#cuobjdump)
tool provided in the CUDA toolkit . For example to extract PTX code, assuming a
CUDA fat binary named `extracted_main.0`:

```
cuobjdump --dump-ptx extracted_main.0
```

The CUDA fat binaries also contain Nvidia SASS, which is generated for a
specific GPU architecture, as opposed to PTX which is generic. And it can be
extracted as follows:

```
cuobjdump -sass extraced_main.0
```

Note that by default this will generate SASS for `sm_50`, the default DPC++ sm
version.  If you want to generate SASS for `sm_XX` you must compile the
original object code using `-Xsycl-target-backend --cuda-gpu-arch=sm_XX`.


#### HIP fat binaries

HIP fat binaries are generated by clang and so the data inside can be extracted
by clang tools, for example assuming an AMD fat binary named `extracted_main.0`,
containing `gfx908` code, the extraction command would be:

```
# Extract specific device code from the bundle
$ ./bin/clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx908 --input=extracted_main.0 --output=device_main
```

This will extract the device code into `device_main`, and it can then be
disassembled with `llvm-objdump` as follows:

```
# Disassemble device code
$ ./bin/llvm-objdump -d device_main
```

The offload bundler step requires specifying the correct target to unbundle,
this can be found by looking into the bundle file, it is a binary but the
targets are in it as plain text, for example, using the same files as in the
previous example:

```
$ strings extracted_main.0 | head -n 3
__CLANG_OFFLOAD_BUNDLE__
host-x86_64-unknown-linux
hipv4-amdgcn-amd-amdhsa--gfx908
```

In that case two binaries are present in the bundle and the `--targets` flag of
the offload bundler to take either one of these triples.
