
# Set default PI HIP platform to AMD
set(SYCL_BUILD_PI_HIP_PLATFORM "AMD" CACHE STRING "PI HIP platform, AMD or NVIDIA")

message(STATUS "Including the PI API HIP backend for ${SYCL_BUILD_PI_HIP_PLATFORM}.")

# Set default ROCm installation directory
set(SYCL_BUILD_PI_HIP_ROCM_DIR "/opt/rocm" CACHE STRING "ROCm installation dir")

# Set HIP include and lib dirs
set(SYCL_BUILD_PI_HIP_INCLUDE_DIR "" CACHE STRING "Override HIP include dir path (set to \"\" for default behavior)")
set(SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR "" CACHE STRING "Override HSA include dir path (set to \"\" for default behavior)")

if(SYCL_ENABLE_KERNEL_FUSION)
  set(SYCL_ENABLE_COMGR ON)
else(SYCL_ENABLE_KERNEL_FUSION)
  set(SYCL_ENABLE_COMGR OFF)
endif(SYCL_ENABLE_KERNEL_FUSION)

if("${SYCL_BUILD_PI_HIP_INCLUDE_DIR}" STREQUAL "")
  set(PI_HIP_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_ROCM_DIR}/include")
else()
  set(PI_HIP_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_INCLUDE_DIR}")
endif()

if("${SYCL_BUILD_PI_HIP_HSA_INCLUDE_DIR}" STREQUAL "")
  # pre v6 versions of ROCM prefix their include directory with /hsa but this
  # was fixed in v6 to act like a well-behaved package
  foreach (SUF hsa/include include)
    if (EXISTS "${SYCL_BUILD_PI_HIP_ROCM_DIR}/${SUF}")
      set(PI_HIP_HSA_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_ROCM_DIR}/${SUF}")
      break()
    endif()
  endforeach()
else()
  set(PI_HIP_HSA_INCLUDE_DIR "${SYCL_BUILD_PI_HIP_INCLUDE_DIR}")
endif()

# Set HIP lib dir
set(SYCL_BUILD_PI_HIP_LIB_DIR "" CACHE STRING "Override HIP lib dir path (set to \"\" for default behavior)")
if("${SYCL_BUILD_PI_HIP_LIB_DIR}" STREQUAL "")
  set(PI_HIP_LIB_DIR "${SYCL_BUILD_PI_HIP_ROCM_DIR}/lib")
else()
  set(PI_HIP_LIB_DIR "${SYCL_BUILD_PI_HIP_LIB_DIR}")
endif()

# Set up defaults for UR
set(UR_HIP_INCLUDE_DIR "${PI_HIP_INCLUDE_DIR}" CACHE PATH
  "Custom ROCm HIP include dir")
set(UR_HIP_HSA_INCLUDE_DIRS "${PI_HIP_HSA_INCLUDE_DIR}" CACHE PATH
  "Custom ROCm HSA include dir")
set(UR_HIP_LIB_DIR "${PI_HIP_LIB_DIR}" CACHE PATH
  "Custom ROCm HIP library dir")

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

# Get the HIP sources so they can be shared with HIP PI plugin
get_target_property(UR_HIP_ADAPTER_SOURCES ur_adapter_hip SOURCES)

# Create pi_hip library
add_sycl_plugin(hip
  SOURCES
    # Some code is shared with the UR adapter
    "../unified_runtime/pi2ur.hpp"
    ${UR_HIP_ADAPTER_SOURCES}
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${sycl_inc_dir}/sycl/detail/pi.hpp"
    "pi_hip.hpp"
    "pi_hip.cpp"
  INCLUDE_DIRS
    ${sycl_plugin_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime
    ${UNIFIED_RUNTIME_SOURCE_DIR}/source/ # for adapters/hip
  LIBRARIES
    UnifiedRuntime-Headers
    UnifiedRuntimeCommon
    UnifiedMemoryFramework
  HEADER
    ${CMAKE_CURRENT_SOURCE_DIR}/include/features.hpp
)
set_target_properties(pi_hip PROPERTIES LINKER_LANGUAGE CXX)

if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")

  set_target_properties(
    rocmdrv PROPERTIES
      IMPORTED_LOCATION                    "${PI_HIP_LIB_DIR}/libamdhip64.so"
      INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
  )
  target_link_libraries(pi_hip PUBLIC rocmdrv)

  if(SYCL_ENABLE_COMGR)
    set(PI_COMGR_VERSION5_HEADER "${PI_HIP_INCLUDE_DIR}/amd_comgr/amd_comgr.h")
    set(PI_COMGR_VERSION4_HEADER "${PI_HIP_INCLUDE_DIR}/amd_comgr.h")
    # The COMGR header changed location between ROCm version 4 and 5.
    # Check for the existence in the version 5 location or fallback to version 4
    if(NOT EXISTS "${PI_COMGR_VERSION5_HEADER}")
      if(NOT EXISTS "${PI_COMGR_VERSION4_HEADER}")
        message(FATAL_ERROR "Could not find AMD COMGR header at "
                            "${PI_COMGR_VERSION5_HEADER} or "
                            "${PI_COMGR_VERSION4_HEADER}, "
                            "check ROCm installation")
      else()
        target_compile_definitions(pi_hip PRIVATE UR_COMGR_VERSION4_INCLUDE)
      endif()
    endif()  

    add_library(amd_comgr SHARED IMPORTED GLOBAL)
    set_target_properties(
      amd_comgr PROPERTIES
        IMPORTED_LOCATION                    "${PI_HIP_LIB_DIR}/libamd_comgr.so"
        INTERFACE_INCLUDE_DIRECTORIES        "${HIP_HEADERS}"
        INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${HIP_HEADERS}"
    )
    target_link_libraries(pi_hip PUBLIC amd_comgr)
    target_compile_definitions(pi_hip PRIVATE SYCL_ENABLE_KERNEL_FUSION)
  endif(SYCL_ENABLE_COMGR)

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

