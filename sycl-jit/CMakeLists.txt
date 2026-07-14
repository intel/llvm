cmake_minimum_required(VERSION 3.20.0 FATAL_ERROR)

# Define a variable holding the root directory of the JIT compiler project
# for use in includes etc.
set(SYCL_JIT_BASE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# For some reason, the LLVMSPIRVLib does not define any public includes.
# To link against the library, define the following link to its include
# directories, similar to how clang/CMakeLists.txt does it.
set(LLVM_SPIRV_INCLUDE_DIRS "${LLVM_MAIN_SRC_DIR}/../llvm-spirv/include")

option(SYCL_JIT_ENABLE_WERROR "Treat all warnings as errors in SYCL JIT library" ON)

# Device targets the JIT compiler supports at runtime. This drives both the
# `JIT_SUPPORT_*` codegen defines and which device libraries (libclc/libspirv)
# are embedded into `libsycl-jit`. Keeping this smaller than the full LLVM
# target list lets a package (e.g. SPIR-V only) avoid embedding libclc, which
# is only needed for the NVPTX/AMDGPU backends and accounts for the bulk of the
# embedded resources.
#
# Valid entries use LLVM target names: SPIRV, NVPTX, AMDGPU. Every entry must
# also be present in LLVM_TARGETS_TO_BUILD so the linked LLVM/Clang libraries
# actually support it.
#
# The default preserves historical behavior: SPIR-V is always supported, and
# NVPTX/AMDGPU are enabled whenever their backends are built into the tree.
set(_sycl_jit_default_targets "SPIRV")
if("NVPTX" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND _sycl_jit_default_targets "NVPTX")
endif()
if("AMDGPU" IN_LIST LLVM_TARGETS_TO_BUILD)
  list(APPEND _sycl_jit_default_targets "AMDGPU")
endif()

set(SYCL_JIT_TARGETS "${_sycl_jit_default_targets}" CACHE STRING
    "Semicolon-separated device targets the JIT compiler supports \
(subset of LLVM_TARGETS_TO_BUILD; valid entries: SPIRV;NVPTX;AMDGPU).")

set(_sycl_jit_valid_targets "SPIRV" "NVPTX" "AMDGPU")
foreach(_jt IN LISTS SYCL_JIT_TARGETS)
  if(NOT _jt IN_LIST _sycl_jit_valid_targets)
    message(FATAL_ERROR
      "SYCL_JIT_TARGETS contains unsupported entry '${_jt}'. "
      "Valid entries are: ${_sycl_jit_valid_targets}.")
  endif()
  if(NOT _jt IN_LIST LLVM_TARGETS_TO_BUILD)
    message(FATAL_ERROR
      "SYCL_JIT_TARGETS entry '${_jt}' is not in LLVM_TARGETS_TO_BUILD "
      "(${LLVM_TARGETS_TO_BUILD}). Add it to LLVM_TARGETS_TO_BUILD first, or "
      "remove it from SYCL_JIT_TARGETS.")
  endif()
endforeach()
message(STATUS "SYCL JIT compiler device targets: ${SYCL_JIT_TARGETS}")

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
