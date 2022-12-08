cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Define a variable holding the root directory of the JIT compiler project
# for use in includes etc.
set(SYCL_JIT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# For some reason, the LLVMSPIRVLib does not define any public includes.
# To link against the library, define the following link to its include
# directories, similar to how clang/CMakeLists.txt does it.
set(LLVM_SPIRV_INCLUDE_DIRS "${LLVM_MAIN_SRC_DIR}/../llvm-spirv/include")

add_subdirectory(jit-compiler)
add_subdirectory(passes)
add_subdirectory(test)
