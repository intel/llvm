message(STATUS "Including the PI API CUDA backend.")

 # cannot rely on cmake support for CUDA; it assumes runtime API is being used.
 # we only require the CUDA driver API to be used
 # CUDA_CUDA_LIBRARY variable defines the path to libcuda.so, the CUDA Driver API library.

find_package(CUDAToolkit 10.1 REQUIRED)

message(STATUS "Including the PI API CUDA backend.")

# Make imported library global to use it within the project.
add_library(cudadrv SHARED IMPORTED GLOBAL)

if (WIN32)
  set_target_properties(
    cudadrv PROPERTIES 
      IMPORTED_IMPLIB               ${CUDA_cuda_driver_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${CUDAToolkit_INCLUDE_DIRS}
  )
else()
  set_target_properties(
    cudadrv PROPERTIES 
      IMPORTED_LOCATION             ${CUDA_cuda_driver_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${CUDAToolkit_INCLUDE_DIRS}
  )
endif()

if (SYCL_ENABLE_XPTI_TRACING)
  set(XPTI_PROXY_SRC "${CMAKE_SOURCE_DIR}/../xpti/src/xpti_proxy.cpp")
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
    "${CUDA_TOOLKIT_ROOT_DIR}/extras/CUPTI/include"
)

set_target_properties(pi_cuda PROPERTIES LINKER_LANGUAGE CXX)

