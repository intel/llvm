# DPC++ end-to-end tests for features under development

We require to have in-tree LIT tests independent from HW (e.g. GPU,
FPGA, etc) and external software (e.g OpenCL, Level Zero, CUDA runtimes, etc).

However, it might be helpful to have E2E tests stored in-tree for features
under active development if atomic change is required for the test and product.
This directory can contain such tests temporarily.

It is developer responsibility to move the tests from this directory to
[DPC++ E2E test suite](https://github.com/intel/llvm-test-suite/tree/intel/SYCL)
or [KhronosGroup/SYCL-CTS](https://github.com/KhronosGroup/SYCL-CTS) once the
feature is stabilized.

# Running tests on deployed DPC++ compiler and runtime
The tests in this directory can be run on deployed DPC++ compiler and runtime.
It is expected that get_device_count_by_type tool is prebuilt by the developer
and tools with full path os provided as value for GET_DEVICE_TOOL LIT
parameter. 

```bash
# clone llvm_project repo
git clone https://github.com/intel/llvm sycl_lit
cd sycl_lit
export ROOT=`pwd`
mkdir build
cd build

# set extra environment
export EXTCMPLRROOT=# path to deployed SYCL compiler and runtime
export GET_DEVICE_TOOL=# Path to the get_device_count_by_type tool
# The get_device_count_by_type tool should be built from sources [under](../../tools/get_device_count_by_type.cpp)
export LEVEL_ZERO_INCLUDE_DIR=# Path to Level_Zero headers (optional)

# Configure project
cmake -DLLVM_TARGETS_TO_BUILD="X86" -DLLVM_EXTERNAL_PROJECTS=sycl-test \
      -DLLVM_EXTERNAL_SYCL_TEST_SOURCE_DIR=$ROOT/sycl/test/on-device \
      -DSYCL_SOURCE_DIR=$ROOT/sycl -DOpenCL_LIBRARIES=$EXTCMPLRROOT/lib \
      $ROOT/llvm

# Build LIT tools
make FileCheck

# Run tests for OpenCL BE
python3 $ROOT/llvm/utils/lit/lit.py -v --param SYCL_PLUGIN=opencl \
                        --param SYCL_TOOLS_DIR="$EXTCMPLRROOT/bin" \
                        --param SYCL_INCLUDE="$EXTCMPLRROOT/include/sycl" \
                        --param SYCL_LIBS_DIR="$EXTCMPLRROOT/lib" \
                        --param GET_DEVICE_TOOL=$GET_DEVICE_TOOL \
                        --param LEVEL_ZERO_INCLUDE_DIR=$L0_HEADER_PATH \
                        tools/sycl-test/

# Run tests for Level_Zero BE
python3 $ROOT/llvm/utils/lit/lit.py -v --param SYCL_PLUGIN=level_zero \
                        --param SYCL_TOOLS_DIR="$EXTCMPLRROOT/bin" \
                        --param SYCL_INCLUDE="$EXTCMPLRROOT/include/sycl" \
                        --param SYCL_LIBS_DIR="$EXTCMPLRROOT/lib" \
                        --param GET_DEVICE_TOOL=$GET_DEVICE_TOOL \
                        --param LEVEL_ZERO_INCLUDE_DIR=$L0_HEADER_PATH \
                        tools/sycl-test/

# Run tests for CUDA BE (if compiler build supports it)
python3 $ROOT/llvm/utils/lit/lit.py -v --param SYCL_PLUGIN=cuda \
                        --param SYCL_TOOLS_DIR="$EXTCMPLRROOT/bin" \
                        --param SYCL_INCLUDE="$EXTCMPLRROOT/include/sycl" \
                        --param SYCL_LIBS_DIR="$EXTCMPLRROOT/lib" \
                        --param GET_DEVICE_TOOL=$GET_DEVICE_TOOL \
                        --param LEVEL_ZERO_INCLUDE_DIR=$L0_HEADER_PATH \
                        tools/sycl-test
```
