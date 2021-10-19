# Enable new IN_LIST operator.
cmake_policy(SET CMP0057 NEW)

set(LLVM_DISTRIBUTIONS
  Toolchain
  Development
  CACHE STRING ""
)

set(COMMON_SYCL_COMPONENTS
  clang
  clang-offload-wrapper
  clang-offload-bundler
  clang-offload-deps
  clang-offload-extract
  level-zero-loader
  level-zero-headers
  sycl-ls
  clang-resource-headers
  OpenCL-Headers
  opencl-aot
  sycl-headers
  sycl-headers-extras
  sycl
  pi_opencl
  pi_level_zero
  libsycldevice
)

set(COMMON_SYCL_TOOLS
  append-file
  file-table-tform
  llc
  llvm-ar
  llvm-foreach
  llvm-no-spir-kernel
  llvm-spirv
  llvm-link
  llvm-objcopy
  sycl-post-link
)

if (SYCL_ENABLE_XPTI_TRACING)
  list(APPEND COMMON_SYCL_COMPONENTS xpti xptifw)
endif()

if (SYCL_BUILD_PI_CUDA)
  list(APPEND COMMON_SYCL_COMPONENTS libspirv-builtins pi_cuda)
endif()

if (SYCL_BUILD_PI_HIP)
  list(APPEND COMMON_SYCL_COMPONENTS libspirv-builtins pi_hip)
endif()

if (NOT MSVC)
  if (SYCL_BUILD_PI_ESIMD_EMULATOR)
    list(APPEND COMMON_SYCL_COMPONENTS pi_esimd_emulator libcmrt-headers)
    list(APPEND COMMON_SYCL_COMPONENTS libcmrt-sos)
  endif()
endif()

set(LLVM_ENABLE_IDE OFF CACHE BOOL "" FORCE)

set(LLVM_Toolchain_TOOLCHAIN_TOOLS
  ${COMMON_SYCL_TOOLS}

  CACHE STRING ""
  )

set(LLVM_Toolchain_DISTRIBUTION_COMPONENTS
  ${COMMON_SYCL_COMPONENTS}
  ${COMMON_SYCL_TOOLS}

  CACHE STRING ""
)

set(LLVM_INSTALL_UTILS ON CACHE BOOL "")

set(EXTRA_TOOLS)
if ("clang-tools-extra" IN_LIST LLVM_ENABLE_PROJECTS)
  list(APPEND EXTRA_TOOLS clang-tidy clang-format)
endif()

set(LLVM_Development_TOOLCHAIN_TOOLS
  FileCheck
  not
  llvm-config
  llvm-cxxdump
  llvm-readobj

  CACHE STRING ""
  )

set(LLVM_Development_DISTRIBUTION_COMPONENTS
  ${LLVM_Development_TOOLCHAIN_TOOLS}
  ${EXTRA_TOOLS}
  CACHE STRING ""
)
