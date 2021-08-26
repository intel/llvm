# Set default PI ROCm platform to AMD
set(SYCL_BUILD_PI_ROCM_PLATFORM "AMD" CACHE STRING "PI ROCm platform, AMD or NVIDIA")

message(STATUS "Including the PI API ROCM backend for ${SYCL_BUILD_PI_ROCM_PLATFORM}.")

# Set default ROCm include dirs
set(SYCL_BUILD_PI_ROCM_INCLUDE_DIR "/opt/rocm/hip/include" CACHE STRING "HIP include dir")
set(SYCL_BUILD_PI_ROCM_HSA_INCLUDE_DIR "/opt/rocm/hsa/include" CACHE STRING "HSA include dir")
set(HIP_HEADERS "${SYCL_BUILD_PI_ROCM_INCLUDE_DIR};${SYCL_BUILD_PI_ROCM_HSA_INCLUDE_DIR}")

# Create pi_rocm library
add_library(pi_rocm SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "${sycl_inc_dir}/CL/sycl/detail/pi.hpp"
  "pi_rocm.hpp"
  "pi_rocm.cpp"
)
add_dependencies(sycl-toolchain pi_rocm)
set_target_properties(pi_rocm PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(pi_rocm PUBLIC OpenCL-Headers)

# Setup include directories
target_include_directories(pi_rocm
  PRIVATE
  ${sycl_inc_dir}
  ${sycl_plugin_dir}
)

if("${SYCL_BUILD_PI_ROCM_PLATFORM}" STREQUAL "AMD")
  # Import HIP runtime library
  set(SYCL_BUILD_PI_ROCM_AMD_LIBRARY "/opt/rocm/hip/lib/libamdhip64.so" CACHE STRING "HIP AMD runtime library")
  add_library(rocmdrv SHARED IMPORTED GLOBAL)

  set_target_properties(
    rocmdrv PROPERTIES
      IMPORTED_LOCATION                    ${SYCL_BUILD_PI_ROCM_AMD_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
  )
  target_link_libraries(pi_rocm PUBLIC rocmdrv)

  # Set HIP define to select AMD platform
  target_compile_definitions(pi_rocm PRIVATE __HIP_PLATFORM_AMD__)

  # Make sure lld is built as part of the toolchain
  add_dependencies(sycl-toolchain lld)
 elseif("${SYCL_BUILD_PI_ROCM_PLATFORM}" STREQUAL "NVIDIA")
  # Import CUDA libraries
  find_package(CUDA REQUIRED)
  list(APPEND HIP_HEADERS ${CUDA_INCLUDE_DIRS})

  # cudadrv may be defined by the CUDA plugin
  if(NOT TARGET cudadrv)
    add_library(cudadrv SHARED IMPORTED GLOBAL)
    set_target_properties(
      cudadrv PROPERTIES
        IMPORTED_LOCATION                    ${CUDA_CUDA_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
    )
  endif()

  add_library(cudart SHARED IMPORTED GLOBAL)
  set_target_properties(
    cudart PROPERTIES
      IMPORTED_LOCATION                    ${CUDA_CUDART_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
  )
  target_link_libraries(pi_rocm PUBLIC cudadrv cudart)

  # Set HIP define to select NVIDIA platform
  target_compile_definitions(pi_rocm PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
  message(FATAL_ERROR "Unspecified PI ROCM platform please set SYCL_BUILD_PI_ROCM_PLATFORM to 'AMD' or 'NVIDIA'")
endif()

add_common_options(pi_rocm)

install(TARGETS pi_rocm
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_rocm
  RUNTIME DESTINATION "bin" COMPONENT pi_rocm
)
