message(STATUS "Including the PI API CUDA backend.")

 # cannot rely on cmake support for CUDA; it assumes runtime API is being used.
 # we only require the CUDA driver API to be used
 # CUDA_CUDA_LIBRARY variable defines the path to libcuda.so, the CUDA Driver API library.

find_package(CUDA 10.0 REQUIRED)

add_library(cudadrv SHARED IMPORTED)

set_target_properties(
  cudadrv PROPERTIES 
    IMPORTED_LOCATION             ${CUDA_CUDA_LIBRARY}
    INTERFACE_INCLUDE_DIRECTORIES ${CUDA_INCLUDE_DIRS}
)

add_library(pi_cuda SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "${sycl_inc_dir}/CL/sycl/detail/pi.hpp"
  "pi_cuda.hpp"
  "pi_cuda.cpp"
)

add_dependencies(pi_cuda
  ocl-headers
)

add_dependencies(sycl-toolchain pi_cuda)

set_target_properties(pi_cuda PROPERTIES LINKER_LANGUAGE CXX)

target_include_directories(pi_cuda PRIVATE "${sycl_inc_dir}")

target_include_directories(pi_cuda INTERFACE ${CUDA_INCLUDE_DIRS})

target_link_libraries(pi_cuda PUBLIC OpenCL-Headers cudadrv)

target_link_libraries(sycl INTERFACE pi_cuda)

add_common_options(pi_cuda)

target_compile_definitions(
  sycl PUBLIC USE_PI_CUDA
)

install(TARGETS pi_cuda
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_cuda
  RUNTIME DESTINATION "bin" COMPONENT pi_cuda
)
