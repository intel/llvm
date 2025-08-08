## End-to-end code generation example

This example demonstrates an end-to-end pipeline of code generation and execution that typically occurs in a JIT engine.
It creates a simple function (adding +1 to an array and storing the result in a second one) via llvm API,
converts it to spirv, and submits it to the runtime.

## Running the example

The following commands are executed from the project root directory.

### Using conda environment

```
$ conda env create -f third_party/deps.yml
$ conda activate examples
$ mkdir build && cd build
# configure with:
$ cmake .. -DUR_BUILD_ADAPTER_L0=ON -DUR_BUILD_EXAMPLE_CODEGEN=ON
$ make
$ ./bin/codegen
```

### Without using conda

To run the example with llvm 13, you will need to make sure that `llvm-13` and `libllvmspirvlib-13-dev` are installed.  

**NOTE**: The example could be working with other llvm versions but it was tested with version 13.

```
$ mkdir build && cd build
# configure with:
$ cmake .. -DUR_BUILD_ADAPTER_L0=ON -DUR_BUILD_EXAMPLE_CODEGEN=ON -DLLVM_DIR=/usr/lib/llvm-13/cmake
$ make
$ ./bin/codegen
```
