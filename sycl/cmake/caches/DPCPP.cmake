# Default configuration of the DPC++ toolchain.
#
# Usage:
#   cmake -G Ninja -C <src>/sycl/cmake/caches/DPCPP.cmake <src>/llvm
#
# Any value can be overridden with -D on the command line. buildbot/configure.py
# loads this file and passes -D only for options that differ from the defaults.
# These defaults assume a native build. CMake loads this file before
# initializing the target platform, so the checks use the host platform.

get_filename_component(DPCPP_SRC_DIR "${CMAKE_CURRENT_LIST_DIR}/../../.." ABSOLUTE)

# Projects.
set(dpcpp_external_projects sycl llvm-spirv opencl xpti xptifw)
set(dpcpp_enable_projects clang)
set(dpcpp_backends opencl)
set(dpcpp_runtime_targets default)

# libdevice build requires a working SYCL toolchain, which is not the case
# with macOS target right now. The same applies to sycl-jit.
if(NOT CMAKE_HOST_APPLE)
  list(APPEND dpcpp_external_projects libdevice)
  list(APPEND dpcpp_backends level_zero level_zero_v2)
  list(APPEND dpcpp_runtime_targets spirv64-unknown-unknown)
  set(SYCL_ENABLE_EXTENSION_JIT ON CACHE BOOL "")
else()
  set(SYCL_ENABLE_EXTENSION_JIT OFF CACHE BOOL "")
endif()

# Respects SYCL_ENABLE_EXTENSION_JIT set before this file is loaded.
if(SYCL_ENABLE_EXTENSION_JIT)
  list(APPEND dpcpp_external_projects sycl-jit)
endif()

# lld is needed on Windows.
if(CMAKE_HOST_WIN32)
  list(APPEND dpcpp_enable_projects lld)
endif()

list(APPEND dpcpp_enable_projects ${dpcpp_external_projects})

set(CMAKE_BUILD_TYPE Release CACHE STRING "")
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "")
set(BUG_REPORT_URL "https://github.com/intel/llvm/issues" CACHE STRING "")

set(LLVM_TARGETS_TO_BUILD "host;SPIRV" CACHE STRING "")
set(LLVM_ENABLE_PROJECTS "${dpcpp_enable_projects}" CACHE STRING "")
set(LLVM_EXTERNAL_PROJECTS "${dpcpp_external_projects}" CACHE STRING "")
set(LLVM_RUNTIME_TARGETS "${dpcpp_runtime_targets}" CACHE STRING "")

set(LLVM_EXTERNAL_SYCL_SOURCE_DIR "${DPCPP_SRC_DIR}/sycl" CACHE PATH "")
set(LLVM_EXTERNAL_LLVM_SPIRV_SOURCE_DIR "${DPCPP_SRC_DIR}/llvm-spirv" CACHE PATH "")
set(LLVM_EXTERNAL_XPTI_SOURCE_DIR "${DPCPP_SRC_DIR}/xpti" CACHE PATH "")
set(XPTI_SOURCE_DIR "${DPCPP_SRC_DIR}/xpti" CACHE PATH "")
set(LLVM_EXTERNAL_XPTIFW_SOURCE_DIR "${DPCPP_SRC_DIR}/xptifw" CACHE PATH "")
set(LLVM_EXTERNAL_LIBDEVICE_SOURCE_DIR "${DPCPP_SRC_DIR}/libdevice" CACHE PATH "")
set(LLVM_EXTERNAL_SYCL_JIT_SOURCE_DIR "${DPCPP_SRC_DIR}/sycl-jit" CACHE PATH "")

set(LLVM_ENABLE_ASSERTIONS ON CACHE BOOL "")
set(LLVM_BUILD_TOOLS ON CACHE BOOL "")
set(LLVM_ENABLE_ZSTD ON CACHE STRING "")
set(LLVM_USE_STATIC_ZSTD ON CACHE BOOL "")
set(LLVM_ENABLE_DOXYGEN OFF CACHE BOOL "")
set(LLVM_ENABLE_SPHINX OFF CACHE BOOL "")
set(LLVM_ENABLE_LLD OFF CACHE BOOL "")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "")
set(LLVM_LINK_LLVM_DYLIB OFF CACHE BOOL "")
set(CLANG_LINK_CLANG_DYLIB OFF CACHE BOOL "")
set(LLVM_SPIRV_ENABLE_LIBSPIRV_DIS OFF CACHE BOOL "")

set(SYCL_INCLUDE_TESTS ON CACHE BOOL "") # Explicitly include all kinds of SYCL tests.
set(SYCL_ENABLE_BACKENDS "${dpcpp_backends}" CACHE STRING "")
set(SYCL_ENABLE_MAJOR_RELEASE_PREVIEW_LIB ON CACHE BOOL "")
set(SYCL_ENABLE_XPTI_TRACING ON CACHE BOOL "")
set(SYCL_ENABLE_WERROR OFF CACHE BOOL "")
set(SYCL_BUILD_UR_HIP_PLATFORM AMD CACHE STRING "")
set(SYCL_CLANG_EXTRA_FLAGS "" CACHE STRING "")
set(XPTI_ENABLE_WERROR OFF CACHE BOOL "")

if(NOT CMAKE_HOST_APPLE)
  set(RUNTIMES_spirv64-unknown-unknown_LLVM_ENABLE_RUNTIMES "compiler-rt;libc" CACHE STRING "")
  set(RUNTIMES_spirv64-unknown-unknown_COMPILER_RT_SANITIZERS_TO_BUILD "" CACHE STRING "")
  # Disable compiler-rt tests (e.g. builtins) for the spirv64 runtime target so
  # they are not added to `check-all`.
  set(RUNTIMES_spirv64-unknown-unknown_COMPILER_RT_INCLUDE_TESTS OFF CACHE BOOL "")
endif()
