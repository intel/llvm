
# Set default PI HIP platform to AMD
set(SYCL_BUILD_PI_HIP_PLATFORM "AMD" CACHE STRING "PI HIP platform, AMD or NVIDIA")

message(STATUS "Including the PI API HIP backend for ${SYCL_BUILD_PI_HIP_PLATFORM}.")

# Set default ROCm installation directory
set(SYCL_BUILD_PI_HIP_ROCM_DIR "/opt/rocm" CACHE STRING "ROCm installation dir")

if(NOT EXISTS "${SYCL_BUILD_PI_HIP_ROCM_DIR}")
  message(FATAL_ERROR "Couldn't find ROCm installation in '${SYCL_BUILD_PI_HIP_ROCM_DIR}',"
                      " please set SYCL_BUILD_PI_HIP_ROCM_DIR to the path of the ROCm installation.")
endif()

# Set HIP include dirs
set(HIP_HEADERS "${SYCL_BUILD_PI_HIP_ROCM_DIR}/hip/include;${SYCL_BUILD_PI_HIP_ROCM_DIR}/hsa/include")

# Create pi_hip library
add_sycl_plugin(hip
  SOURCES
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${sycl_inc_dir}/sycl/detail/pi.hpp"
    "pi_hip.hpp"
    "pi_hip.cpp"
  INCLUDE_DIRS
    ${sycl_plugin_dir}
  HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/features.hpp
)
set_target_properties(pi_hip PROPERTIES LINKER_LANGUAGE CXX)

if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
  # Import HIP runtime library
  add_library(rocmdrv SHARED IMPORTED GLOBAL)

  set_target_properties(
    rocmdrv PROPERTIES
      IMPORTED_LOCATION                    "${SYCL_BUILD_PI_HIP_ROCM_DIR}/hip/lib/libamdhip64.so"
      INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
  )
  target_link_libraries(pi_hip PUBLIC rocmdrv)

  # Set HIP define to select AMD platform
  target_compile_definitions(pi_hip PRIVATE __HIP_PLATFORM_AMD__)
elseif("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "NVIDIA")
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
  target_link_libraries(pi_hip PUBLIC cudadrv cudart)

  # Set HIP define to select NVIDIA platform
  target_compile_definitions(pi_hip PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
  message(FATAL_ERROR "Unspecified PI HIP platform please set SYCL_BUILD_PI_HIP_PLATFORM to 'AMD' or 'NVIDIA'")
endif()

