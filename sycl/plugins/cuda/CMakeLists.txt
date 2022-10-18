message(STATUS "Including the PI API CUDA backend.")

 # cannot rely on cmake support for CUDA; it assumes runtime API is being used.
 # we only require the CUDA driver API to be used
 # CUDA_CUDA_LIBRARY variable defines the path to libcuda.so, the CUDA Driver API library.

find_package(CUDA 10.1 REQUIRED)

# Make imported library global to use it within the project.
add_library(cudadrv SHARED IMPORTED GLOBAL)

if (WIN32)
  set_target_properties(
    cudadrv PROPERTIES 
      IMPORTED_IMPLIB               ${CUDA_CUDA_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
  )
else()
  set_target_properties(
    cudadrv PROPERTIES 
      IMPORTED_LOCATION             ${CUDA_CUDA_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
  )
endif()

if (SYCL_ENABLE_XPTI_TRACING)
  set(XPTI_PROXY_SRC "${CMAKE_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp")
endif()

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

add_sycl_plugin(cuda
  SOURCES
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${sycl_inc_dir}/sycl/detail/pi.hpp"
    "pi_cuda.hpp"
    "pi_cuda.cpp"
    "tracing.cpp"
    ${XPTI_PROXY_SRC}
  LIBRARIES
    cudadrv
    ${CUDA_cupti_LIBRARY}
)

if (SYCL_ENABLE_XPTI_TRACING)
  target_compile_definitions(pi_cuda PRIVATE
    XPTI_ENABLE_INSTRUMENTATION
    XPTI_STATIC_LIBRARY
  )
  target_include_directories(pi_cuda PRIVATE "${CMAKE_SOURCE_DIR}/../xpti/include")
  target_link_libraries(pi_cuda PRIVATE ${CMAKE_DL_LIBS})
endif()


target_include_directories(pi_cuda
  PRIVATE
    ${sycl_inc_dir}
    ${CUDA_CUPTI_INCLUDE_DIR}
)

set_target_properties(pi_cuda PROPERTIES LINKER_LANGUAGE CXX)

