# LLVM/SPIR-V Bi-Directional Translator

[![Build Status](https://travis-ci.org/KhronosGroup/SPIRV-LLVM-Translator.svg?branch=master)](https://travis-ci.org/KhronosGroup/SPIRV-LLVM-Translator)

This repository contains source code for the LLVM/SPIR-V Bi-Directional Translator, a library and tool for translation between LLVM IR and [SPIR-V](https://www.khronos.org/registry/spir-v/).

The LLVM/SPIR-V Bi-Directional Translator is open source software. You may freely distribute it under the terms of the license agreement found in LICENSE.txt.


## Directory Structure


The files/directories related to the translator:

* [include/LLVMSPIRVLib.h](include/LLVMSPIRVLib.h) - header file
* [lib/SPIRV](lib/SPIRV) - library for SPIR-V in-memory representation, decoder/encoder and LLVM/SPIR-V translator
* [tools/llvm-spirv](tools/llvm-spirv) - command line utility for translating between LLVM bitcode and SPIR-V binary

## Build Instructions

The `master` branch of this repo is aimed to be buildable with the latest LLVM `master` or `trunk` revision.

### Build with pre-installed LLVM

The translator can be built with the latest(nightly) package of LLVM. For Ubuntu and Debian systems LLVM provides repositories with nightly builds at http://apt.llvm.org/. For example the latest package for Ubuntu 16.04 can be installed with the following commands:
```
sudo add-apt-repository "deb http://apt.llvm.org/xenial/ llvm-toolchain-xenial main"
sudo apt-get update
sudo apt-get install llvm-10-dev llvm-10-tools clang-10 libclang-10-dev
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
cmake ../llvm
make llvm-spirv -j`nproc`
```

## Test instructions

All tests related to the translator are placed in the [test](test) directory. A number of the tests require spirv-as (part of SPIR-V Tools) to run, but the remainder of the tests can still be run without this. Optionally the tests can make use of spirv-val (part of SPIRV-Tools) in order to validate the generated SPIR-V against the official SPIR-V specification.

In case tests are failing due to SPIRV-Tools not supporting certain SPIR-V features, please get an updated package. The `PKG_CONFIG_PATH` environmental variable can be used to let cmake point to a custom installation.

Execute the following command inside the build directory to run translator tests:
```
make test
```
This requires that the `-DLLVM_INCLUDE_TESTS=ON` and
`-DLLVM_EXTERNAL_LIT="/usr/lib/llvm-10/build/utils/lit/lit.py"` arguments were
passed to CMake during the build step.

The translator test suite can be disabled by passing
`-DLLVM_SPIRV_INCLUDE_TESTS=OFF` to cmake.

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
    * `-spirv-ocl-builtins-version` - to specify target version of OpenCL builtins to translate to (default CL1.2)

3. Other options accepted by `llvm-spirv`

    * `-o file_name` - to specify output name
    * `-spirv-debug` - output debugging information
    * `-spirv-text` - read/write SPIR-V in an internal textual format for debugging purpose. The textual format is not defined by SPIR-V spec.
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

Allowed values are `1.0`/`1.1`.

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

Code on the master branch in this repository is intended to be compatible with master/trunk branch of the [llvm](https://github.com/llvm-mirror/llvm) project. That is, for an OpenCL kernel compiled to llvm bitcode by the latest version(built with the latest git commit or svn revision) of Clang it should be possible to translate it to SPIR-V with the llvm-spirv tool.

All new development should be done on the master branch.

To have versions compatible with released versions of LLVM and Clang, corresponding branches are available in this repository. For example, to build the translator with LLVM 7.0 ([release_70](https://github.com/llvm-mirror/llvm/tree/release_70)) one should use the [llvm_release_70](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/tree/llvm_release_70) branch. As a general rule, commits from the master branch may be backported to the release branches as long as they do not depend on features from a later LLVM/Clang release and there are no objections from the maintainer(s). There is no guarantee that older release branches are proactively kept up to date with master, but you can request certain commits on older release branches by creating a pull request or raising an issue on GitHub.
