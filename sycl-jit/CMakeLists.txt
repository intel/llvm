cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

# Define a variable holding the root directory of the JIT compiler project
# for use in includes etc.
set(SYCL_JIT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# For some reason, the LLVMSPIRVLib does not define any public includes.
# To link against the library, define the following link to its include
# directories, similar to how clang/CMakeLists.txt does it.
set(LLVM_SPIRV_INCLUDE_DIRS "${LLVM_MAIN_SRC_DIR}/../llvm-spirv/include")

option(SYCL_JIT_ENABLE_WERROR "Treat all warnings as errors in SYCL JIT library" ON)

# sycl-jit warning flags; host GCC: -Wno-error=conversion on LLVM/Clang headers.
function(sycl_jit_apply_target_warning_flags target)
  if(WIN32 OR CYGWIN)
    return()
  endif()
  target_compile_options(${target} PRIVATE
    -Wall -Wextra -Wconversion -Wimplicit-fallthrough
    $<$<BOOL:${SYCL_JIT_ENABLE_WERROR}>:-Werror>
    $<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-error=conversion>)
endfunction()

add_subdirectory(jit-compiler)
add_subdirectory(passes)

# Loadable plugins for opt aren't supported on Windows,
# so we can't execute the tests.
if (NOT WIN32 AND NOT CYGWIN)
  add_subdirectory(test)
endif()
