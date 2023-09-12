
# Set default PI HIP platform to AMD
set(SYCL_BUILD_PI_HIP_PLATFORM "AMD" CACHE STRING "PI HIP platform, AMD or NVIDIA")

message(STATUS "Including the PI API HIP backend for ${SYCL_BUILD_PI_HIP_PLATFORM}.")

# Set default ROCm installation directory
set(SYCL_BUILD_PI_HIP_ROCM_DIR "/opt/rocm" CACHE STRING "ROCm installation dir")

# Set HIP include and lib dirs
set(SYCL_BUILD_PI_HIP_INCLUDE_DIR "" CACHE STRING "Override HIP include dir path (set to \"\" for default behavior)")
set(SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR "" CACHE STRING "Override HSA include dir path (set to \"\" for default behavior)")

if("${SYCL_BUILD_PI_HIP_INCLUDE_DIR}" STREQUAL "")
  set(PI_HIP_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_ROCM_DIR}/include")
else()
  set(PI_HIP_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_INCLUDE_DIR}")
endif()

if("${SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR}" STREQUAL "")
  set(PI_HIP_HSA_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_ROCM_DIR}/hsa/include")
else()
  set(PI_HIP_HSA_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_INCLUDE_DIR}")
endif()

# Set HIP lib dir
set(SYCL_BUILD_PI_HIP_LIB_DIR "" CACHE STRING "Override HIP lib dir path (set to \"\" for default behavior)")
if("${SYCL_BUILD_PI_HIP_LIB_DIR}" STREQUAL "")
  set(PI_HIP_LIB_DIR "${SYCL_BUILD_PI_HIP_ROCM_DIR}/hip/lib")
else()
  set(PI_HIP_LIB_DIR "${SYCL_BUILD_PI_HIP_LIB_DIR}")
endif()

# Mark override options for advanced usage
mark_as_advanced(SYCL_BUILD_PI_HIP_INCLUDE_DIR SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR SYCL_BUILD_PI_HIP_LIB_DIR)

# Check if ROCm install paths exists
# N.B. Doesn't check if all override options are set: HSA and HIP include (and HIP lib for AMD platform)
if(("${SYCL_BUILD_PI_HIP_INCLUDE_DIR}" STREQUAL "") OR
   ("${SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR}" STREQUAL "") OR
   (("${SYCL_BUILD_PI_HIP_LIB_DIR}" STREQUAL "") AND ("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")))

  if(NOT EXISTS "${SYCL_BUILD_PI_HIP_ROCM_DIR}")
    message(FATAL_ERROR "Couldn't find ROCm installation in '${SYCL_BUILD_PI_HIP_ROCM_DIR}',"
                        " please set SYCL_BUILD_PI_HIP_ROCM_DIR to the path of the ROCm installation.")
  endif()
endif()

# Check if HIP include path exists
if(NOT EXISTS "${PI_HIP_INCLUDE_DIR}")
  if("${SYCL_BUILD_PI_HIP_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Couldn't find the HIP include directory at '${PI_HIP_INCLUDE_DIR}',"
                        " please check ROCm installation and possibly set SYCL_BUILD_PI_HIP_INCLUDE_DIR to the path of the HIP include directory for non-standard install paths.")
  else()
    message(FATAL_ERROR "Couldn't find the HIP include directory at '${PI_HIP_INCLUDE_DIR}',"
                        " please set SYCL_BUILD_PI_HIP_INCLUDE_DIR to the path of the HIP include directory from the ROCm installation.")
  endif()
endif()

# Check if HSA include path exists
if(NOT EXISTS "${PI_HIP_HSA_INCLUDE_DIR}")
  if("${SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR}" STREQUAL "")
    message(FATAL_ERROR "Couldn't find the HSA include directory at '${PI_HIP_HSA_INCLUDE_DIR}',"
                        " please check ROCm installation and possibly set SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR to the path of the HSA include directory for non-standard install paths.")
  else()
    message(FATAL_ERROR "Couldn't find the HSA include directory at '${PI_HIP_HSA_INCLUDE_DIR}',"
                        " please set SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR to the path of the HSA include directory from the ROCm installation.")
  endif()
endif()

# Check if HIP library path exists (AMD platform only)
if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
  if(NOT EXISTS "${PI_HIP_LIB_DIR}")
    if("${SYCL_BUILD_PI_HIP_LIB_DIR}" STREQUAL "")
      message(FATAL_ERROR "Couldn't find the HIP library directory at '${PI_HIP_LIB_DIR}',"
          " please check ROCm installation and possibly set SYCL_BUILD_PI_HIP_LIB_DIR to the path of the HIP library directory for non-standard install paths.")
    else()
      message(FATAL_ERROR "Couldn't find the HIP library directory at '${PI_HIP_LIB_DIR}',"
                          " please set SYCL_BUILD_PI_HIP_LIB_DIR to the path of the HIP library directory from the ROCm installation.")
    endif()
  endif()
endif()

# Set includes used in added library (rocmdrv)
set(HIP_HEADERS "${PI_HIP_INCLUDE_DIR};${PI_HIP_HSA_INCLUDE_DIR}")

# Create pi_hip library
add_sycl_plugin(hip
  SOURCES
    # Some code is shared with the UR adapter
    "../unified_runtime/pi2ur.hpp"
    "../unified_runtime/pi2ur.cpp"
    "../unified_runtime/ur/ur.hpp"
    "../unified_runtime/ur/ur.cpp"
    "../unified_runtime/ur/adapters/hip/adapter.cpp"
    "../unified_runtime/ur/adapters/hip/adapter.hpp"
    "../unified_runtime/ur/adapters/hip/command_buffer.cpp"
    "../unified_runtime/ur/adapters/hip/command_buffer.hpp"
    "../unified_runtime/ur/adapters/hip/common.cpp"
    "../unified_runtime/ur/adapters/hip/common.hpp"
    "../unified_runtime/ur/adapters/hip/context.cpp"
    "../unified_runtime/ur/adapters/hip/context.hpp"
    "../unified_runtime/ur/adapters/hip/device.cpp"
    "../unified_runtime/ur/adapters/hip/device.hpp"
    "../unified_runtime/ur/adapters/hip/enqueue.cpp"
    "../unified_runtime/ur/adapters/hip/event.cpp"
    "../unified_runtime/ur/adapters/hip/event.hpp"
    "../unified_runtime/ur/adapters/hip/image.cpp"
    "../unified_runtime/ur/adapters/hip/kernel.cpp"
    "../unified_runtime/ur/adapters/hip/kernel.hpp"
    "../unified_runtime/ur/adapters/hip/memory.cpp"
    "../unified_runtime/ur/adapters/hip/memory.hpp"
    "../unified_runtime/ur/adapters/hip/platform.cpp"
    "../unified_runtime/ur/adapters/hip/platform.hpp"
    "../unified_runtime/ur/adapters/hip/program.cpp"
    "../unified_runtime/ur/adapters/hip/program.hpp"
    "../unified_runtime/ur/adapters/hip/queue.cpp"
    "../unified_runtime/ur/adapters/hip/queue.hpp"
    "../unified_runtime/ur/adapters/hip/sampler.cpp"
    "../unified_runtime/ur/adapters/hip/sampler.hpp"
    "../unified_runtime/ur/adapters/hip/ur_interface_loader.cpp"
    "../unified_runtime/ur/adapters/hip/usm.cpp"
    "../unified_runtime/ur/adapters/hip/usm.hpp"
    "../unified_runtime/ur/adapters/hip/usm_p2p.cpp"
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${sycl_inc_dir}/sycl/detail/pi.hpp"
    "pi_hip.hpp"
    "pi_hip.cpp"
  INCLUDE_DIRS
    ${sycl_plugin_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime
  LIBRARIES
    UnifiedRuntime-Headers
    UnifiedRuntimeCommon
    UnifiedMallocFramework
  HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/features.hpp
)
set_target_properties(pi_hip PROPERTIES LINKER_LANGUAGE CXX)

if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
  # Import HIP runtime library
  add_library(rocmdrv SHARED IMPORTED GLOBAL)

  set_target_properties(
    rocmdrv PROPERTIES
      IMPORTED_LOCATION                    "${PI_HIP_LIB_DIR}/libamdhip64.so"
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

