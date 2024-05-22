message(STATUS "Including the PI API CUDA backend.")

if (SYCL_ENABLE_XPTI_TRACING)
  # cannot rely on cmake support for CUDA; it assumes runtime API is being used.
  # we only require the CUDA driver API to be used
  # CUDA_CUDA_LIBRARY variable defines the path to libcuda.so, the CUDA Driver API library.
  find_package(CUDA 10.1 REQUIRED)

  # The following two if's can be removed when FindCUDA -> FindCUDAToolkit.
  # CUDA_CUPTI_INCLUDE_DIR -> CUDAToolkit_CUPTI_INCLUDE_DIR
  include(FindCUDACupti)
  if(NOT CUDA_CUPTI_INCLUDE_DIR)
    find_cuda_cupti_include_dir()
  endif()
  # CUDA_cupti_LIBRARY -> CUDAToolkit_cupti_LIBRARY
  if(NOT CUDA_cupti_LIBRARY)
    find_cuda_cupti_library()
  endif()

  set(XPTI_PROXY_SRC "${CMAKE_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp")
  set(XPTI_INCLUDE
    "${CMAKE_SOURCE_DIR}/../xpti/include"
    "${CUDA_CUPTI_INCLUDE_DIR}"
  )
  set(XPTI_LIBS
    "${CMAKE_DL_LIBS}"
    "${CUDA_cupti_LIBRARY}"
  )
endif()

# Get the CUDA adapter sources so they can be shared with the CUDA PI plugin
get_target_property(UR_CUDA_ADAPTER_SOURCES ur_adapter_cuda SOURCES)

add_sycl_plugin(cuda
  SOURCES
    ${UR_CUDA_ADAPTER_SOURCES}
    # Some code is shared with the UR adapter
    "../unified_runtime/pi2ur.hpp"
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${sycl_inc_dir}/sycl/detail/pi.hpp"
    "pi_cuda.hpp"
    "pi_cuda.cpp"
    ${XPTI_PROXY_SRC}
  INCLUDE_DIRS
    ${sycl_inc_dir}
    ${XPTI_INCLUDE}
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime # for Unified Runtime
    ${UNIFIED_RUNTIME_SOURCE_DIR}/source/ # for adapters/cuda
  LIBRARIES
    cudadrv
    ${XPTI_LIBS}
    UnifiedRuntime-Headers
    UnifiedRuntimeCommon
    UnifiedMemoryFramework
  HEADER "${CMAKE_CURRENT_SOURCE_DIR}/include/features.hpp"
)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(pi_cuda PRIVATE
    XPTI_ENABLE_INSTRUMENTATION
    XPTI_STATIC_LIBRARY
  )
endif()

if(CUDA_cupti_LIBRARY)
  target_compile_definitions(pi_cuda PRIVATE
    "-DCUPTI_LIB_PATH=\"${CUDA_cupti_LIBRARY}\"")
endif()

set_target_properties(pi_cuda PROPERTIES LINKER_LANGUAGE CXX)

