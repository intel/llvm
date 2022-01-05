
# Set default PI HIP platform to AMD
set(SYCL_BUILD_PI_HIP_PLATFORM "AMD" CACHE STRING "PI HIP platform, AMD or NVIDIA")

message(STATUS "Including the PI API HIP backend for ${SYCL_BUILD_PI_HIP_PLATFORM}.")

# Set default HIP include dirs
set(SYCL_BUILD_PI_HIP_INCLUDE_DIR "/opt/rocm/hip/include" CACHE STRING "HIP include dir")
set(SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR "/opt/rocm/hsa/include" CACHE STRING "HSA include dir")
set(HIP_HEADERS "${SYCL_BUILD_PI_HIP_INCLUDE_DIR};${SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR}")

# Create pi_hip library
add_library(pi_hip SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "${sycl_inc_dir}/CL/sycl/detail/pi.hpp"
  "pi_hip.hpp"
  "pi_hip.cpp"
)
add_dependencies(sycl-toolchain pi_hip)
set_target_properties(pi_hip PROPERTIES LINKER_LANGUAGE CXX)
target_link_libraries(pi_hip PUBLIC OpenCL-Headers)

# Setup include directories
target_include_directories(pi_hip
  PRIVATE
  ${sycl_inc_dir}
  ${sycl_plugin_dir}
)

if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
  # Import HIP runtime library
  set(SYCL_BUILD_PI_HIP_AMD_LIBRARY "/opt/rocm/hip/lib/libamdhip64.so" CACHE STRING "HIP AMD runtime library")
  add_library(rocmdrv SHARED IMPORTED GLOBAL)

  set_target_properties(
    rocmdrv PROPERTIES
      IMPORTED_LOCATION                    ${SYCL_BUILD_PI_HIP_AMD_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
  )
  target_link_libraries(pi_hip PUBLIC rocmdrv)

  # Set HIP define to select AMD platform
  target_compile_definitions(pi_hip PRIVATE __HIP_PLATFORM_AMD__)

  # Make sure lld is built as part of the toolchain
  add_dependencies(sycl-toolchain lld)
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

add_common_options(pi_hip)

install(TARGETS pi_hip
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_hip
  RUNTIME DESTINATION "bin" COMPONENT pi_hip
)
