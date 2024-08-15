cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

# Define a variable holding the root directory of the JIT compiler project
# for use in includes etc.
set(SYCL_JIT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# For some reason, the LLVMSPIRVLib does not define any public includes.
# To link against the library, define the following link to its include
# directories, similar to how clang/CMakeLists.txt does it.
set(LLVM_SPIRV_INCLUDE_DIRS "${LLVM_MAIN_SRC_DIR}/../llvm-spirv/include")

# Set library-wide warning options.
set(SYCL_JIT_WARNING_FLAGS -Wall -Wextra)

option(SYCL_JIT_ENABLE_WERROR "Treat all warnings as errors in SYCL kernel JIT library" ON)
if(SYCL_JIT_ENABLE_WERROR)
  list(APPEND SYCL_JIT_WARNING_FLAGS -Werror)
endif(SYCL_JIT_ENABLE_WERROR)

if(WIN32)
  message(WARNING "Kernel JIT not yet supported on Windows")
else(WIN32)
  add_subdirectory(common)
  add_subdirectory(jit-compiler)
  add_subdirectory(passes)
  add_subdirectory(test)
endif(WIN32)
