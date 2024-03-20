# LLVM/SPIR-V Bi-Directional Translator

[![Out-of-tree build & tests](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/actions/workflows/check-out-of-tree-build.yml/badge.svg?branch=main&event=schedule)](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/actions?query=workflow%3A%22Out-of-tree+build+%26+tests%22+event%3Aschedule)
[![In-tree build & tests](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/actions/workflows/check-in-tree-build.yml/badge.svg?branch=main&event=schedule)](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/actions?query=workflow%3A%22In-tree+build+%26+tests%22+event%3Aschedule)

This repository contains source code for the LLVM/SPIR-V Bi-Directional Translator, a library and tool for translation between LLVM IR and [SPIR-V](https://www.khronos.org/registry/spir-v/).
This project currently only supports the OpenCL/compute "flavour" of SPIR-V: it consumes and produces SPIR-V modules that declare the `Kernel` capability.

The LLVM/SPIR-V Bi-Directional Translator is open source software. You may freely distribute it under the terms of the license agreement found in LICENSE.txt.


## Directory Structure


The files/directories related to the translator:

* [include/LLVMSPIRVLib.h](include/LLVMSPIRVLib.h) - header file
* [lib/SPIRV](lib/SPIRV) - library for SPIR-V in-memory representation, decoder/encoder and LLVM/SPIR-V translator
* [tools/llvm-spirv](tools/llvm-spirv) - command line utility for translating between LLVM bitcode and SPIR-V binary

## Build Instructions

The `main` branch of this repo is aimed to be buildable with the latest
LLVM `main` revision.

### Build with pre-installed LLVM

The translator can be built with the latest(nightly) package of LLVM. For Ubuntu and Debian systems LLVM provides repositories with nightly builds at http://apt.llvm.org/. For example the latest package for Ubuntu 16.04 can be installed with the following commands:
```
wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -
sudo add-apt-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main"
sudo apt-get update
sudo apt-get install llvm-19-dev llvm-19-tools clang-19 libclang-19-dev
```
The installed version of LLVM will be used by default for out-of-tree build of the translator.
```
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
mkdir SPIRV-LLVM-Translator/build && cd SPIRV-LLVM-Translator/build
cmake ..
make llvm-spirv -j`nproc`
```

### Build with pre-built LLVM

If you have a custom build (based on the latest version) of LLVM libraries you
can link the translator against it.

```
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
mkdir SPIRV-LLVM-Translator/build && cd SPIRV-LLVM-Translator/build
cmake .. -DLLVM_DIR=<llvm_build_dir>/lib/cmake/llvm/
make llvm-spirv -j`nproc`
```

If the translator is used as part of another CMake project, you will need
to define `LLVM_SPIRV_BUILD_EXTERNAL`:

```
cmake .. -DLLVM_DIR=<llvm_build_dir>/lib/cmake/llvm/ -DLLVM_SPIRV_BUILD_EXTERNAL=YES
```

Where `llvm_build_dir` is the LLVM build directory.

### LLVM in-tree build

The translator can be built as a regular LLVM subproject. To do that you need to clone it into the `llvm/projects` or `llvm/tools` directory.
```
git clone https://github.com/llvm/llvm-project.git
cd llvm-project/llvm/projects
git clone https://github.com/KhronosGroup/SPIRV-LLVM-Translator.git
```
Run (or re-run) cmake as usual for LLVM. After that you should have `llvm-spirv` and `check-llvm-spirv` targets available.
```
mkdir llvm-project/build && cd llvm-project/build
cmake ../llvm -DLLVM_ENABLE_PROJECTS="clang"
make llvm-spirv -j`nproc`
```

Note on enabling the `clang` project: there are tests in the translator that depend
on `clang` binary, which makes clang a required dependency (search for
`LLVM_SPIRV_TEST_DEPS` in [test/CMakeLists.txt](test/CMakeLists.txt)) for
`check-llvm-spirv` target.

Building clang from sources takes time and resources and it can be avoided:
- if you are not interested in launching unit-tests for the translator after
  build, you can disable generation of test targets by passing
  `-DLLVM_SPIRV_INCLUDE_TESTS=OFF` option.
- if you are interested in launching unit-tests, but don't want to build clang
  you can pass `-DSPIRV_SKIP_CLANG_BUILD` cmake option to avoid adding `clang`
  as dependency for `check-llvm-spirv` target. However, LIT will search for
  `clang` binary when tests are launched and it should be available at this
  point.
- building and testing completely without `clang` is not supported at the
  moment, see [KhronosGroup/SPIRV-LLVM-Translator#477](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/issues/477)
  to track progress, discuss and contribute.

### Build with SPIRV-Tools

The translator can use [SPIRV-Tools](https://github.com/KhronosGroup/SPIRV-Tools) to generate assembly with widely adopted syntax.
If SPIRV-Tools have been installed prior to the build it will be detected and
used automatically. However it is also possible to enable use of SPIRV-Tools
from a custom location using the following instructions:

1. Checkout, build and install SPIRV-Tools using
   [the following instructions](https://github.com/KhronosGroup/SPIRV-Tools#build).
   Example using CMake with Ninja:
```
cmake -G Ninja <SPIRV-Tools source location> -DCMAKE_INSTALL_PREFIX=<SPIRV-Tools installation location>
ninja install
```
2. Point pkg-config to the SPIR-V tools installation when configuring the translator by setting
   `PKG_CONFIG_PATH=<SPIRV-Tools installation location>/lib/pkgconfig/` variable
   before the cmake line invocation.
   Example:
```
PKG_CONFIG_PATH=<SPIRV-Tools installation location>/lib/pkgconfig/ cmake <other options>
```

To verify the SPIR-V Tools integration in the translator build, run the following line
```
llvm-spirv --spirv-tools-dis input.bc -o -
```
The output should be printed in the standard assembly syntax.

## Configuring SPIR-V Headers

The translator build is dependent on the official Khronos header file
`spirv.hpp` that maps SPIR-V extensions, decorations, instructions,
etc. onto numeric tokens. The official header version is available at
[KhronosGroup/SPIRV-Headers](https://github.com/KhronosGroup/SPIRV-Headers).
There are several options for accessing the header file:
- By default, the header file repository will be downloaded from
  Khronos Group GitHub and put into `<build_dir>/SPIRV-Headers`.
- If you are building the translator in-tree, you can manually
  download the SPIR-V Headers repo into `llvm/projects` - this
  location will be automatically picked up by the LLVM build
  scripts. Make sure the folder retains its default naming in
  that of `SPIRV-Headers`.
- Any build type can also use an external installation of SPIR-V
  Headers - if you have the headers downloaded somewhere in your
  system and want to use that version, simply extend your CMake
  command with `-DLLVM_EXTERNAL_PROJECTS="SPIRV-Headers"
  -DLLVM_EXTERNAL_SPIRV_HEADERS_SOURCE_DIR=</path/to/headers_dir>`.

## Test instructions

All tests related to the translator are placed in the [test](test) directory. A number of the tests require spirv-as (part of SPIR-V Tools) to run, but the remainder of the tests can still be run without this. Optionally the tests can make use of spirv-val (part of SPIRV-Tools) in order to validate the generated SPIR-V against the official SPIR-V specification.

In case tests are failing due to SPIRV-Tools not supporting certain SPIR-V features, please get an updated package. The `PKG_CONFIG_PATH` environmental variable can be used to let cmake point to a custom installation.

Execute the following command inside the build directory to run translator tests:
```
make test
```
This requires that the `-DLLVM_SPIRV_INCLUDE_TESTS=ON` argument is
passed to CMake during the build step. Additionally,
`-DLLVM_EXTERNAL_LIT="/usr/lib/llvm-19/build/utils/lit/lit.py"` is
needed when building with a pre-installed version of LLVM.

The translator test suite can be disabled by passing
`-DLLVM_SPIRV_INCLUDE_TESTS=OFF` to CMake.

## Run Instructions for `llvm-spirv`


To translate between LLVM IR and SPIR-V:

1. Execute the following command to translate `input.bc` to `input.spv`
    ```
    llvm-spirv input.bc
    ```

2. Execute the following command to translate `input.spv` to `input.bc`
    ```
    llvm-spirv -r input.spv
    ```
    Recommended options:
    * `-spirv-target-env` - to specify target version of OpenCL builtins to translate to (default CL1.2)

3. Other options accepted by `llvm-spirv`

    * `-o file_name` - to specify output name
    * `-spirv-debug` - output debugging information
    * `-spirv-text` - read/write SPIR-V in an internal textual format for debugging purpose. The textual format is not defined by SPIR-V spec.
    * `--spirv-tools-dis` - print SPIR-V assembly in SPIRV-Tools format. Only available on [builds with SPIRV-Tools](#build-with-spirv-tools).
    * `-help` - to see full list of options

Translation from LLVM IR to SPIR-V and then back to LLVM IR is not guaranteed to
produce the original LLVM IR.  In particular, LLVM intrinsic call instructions
may get replaced by function calls to OpenCL builtins and metadata may be
dropped.

### Handling SPIR-V versions generated by the translator

There is one option to control the behavior of the translator with respect to
the version of the SPIR-V file which is being generated/consumed.

* `-spirv-max-version=` - this option allows restricting the
  SPIRV-LLVM-Translator **not** to generate a SPIR-V with a version which is
  higher than the one specified via this option.

  If the `-r` option was also specified, the SPIRV-LLVM-Translator will reject
  the input file and emit an error if the SPIR-V version in it is higher than
  one specified via this option.

Allowed values are `1.0`, `1.1`, `1.2`, `1.3`, and `1.4`.

More information can be found in
[SPIR-V versions and extensions handling](docs/SPIRVVersionsAndExtensionsHandling.rst)

### Handling SPIR-V extensions generated by the translator

By default, during SPIR-V generation, the translator doesn't use any extensions.
However, during SPIR-V consumption, the translator accepts input files that use
any known extensions.

If certain extensions are required to be enabled or disabled, the following
command line option can be used:

* ``--spirv-ext=`` - this options allows controlling which extensions are
  allowed/disallowed

Valid value for this option is comma-separated list of extension names prefixed
with ``+`` or ``-`` - plus means allow to use extension, minus means disallow
to use extension. There is one more special value which can be used as extension
name in this option: ``all`` - it affects all extension which are known to the
translator.

If ``--spirv-ext`` contains the name of an extension which is not known for the
translator, it will emit an error.

More information can be found in
[SPIR-V versions and extensions handling](docs/SPIRVVersionsAndExtensionsHandling.rst)

## Branching strategy

Code on the main branch in this repository is intended to be compatible with
the main branch of the [llvm](https://github.com/llvm/llvm-project)
project. That is, for an OpenCL kernel compiled to llvm bitcode by the latest
git revision of Clang it should be possible to translate it to SPIR-V with the
llvm-spirv tool.

All new development should be done on the main branch.

To have versions compatible with released versions of LLVM and Clang,
corresponding tags are available in this repository. For example, to build
the translator with
[LLVM 7.0.0](https://github.com/llvm/llvm-project/tree/llvmorg-7.0.0)
one should use the
[v7.0.0-1](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/v7.0.0-1)
tag. The 7.x releases are maintained on the
[llvm_release_70](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/llvm_release_70)
branch. As a general rule, commits from the main branch may be backported to
the release branches as long as they do not depend on features from a later
LLVM/Clang release and there are no objections from the maintainer(s). There
is no guarantee that older release branches are proactively kept up to date
with main, but you can request specific commits on older release branches by
creating a pull request or raising an issue on GitHub.
