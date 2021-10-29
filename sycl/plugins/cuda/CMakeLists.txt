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

add_sycl_library(pi_cuda SHARED
  SOURCES
    "${sycl_inc_dir}/CL/sycl/detail/pi.h"
    "${sycl_inc_dir}/CL/sycl/detail/pi.hpp"
    "pi_cuda.hpp"
    "pi_cuda.cpp"
  INCLUDES
    ${sycl_inc_dir}
  LINK
    OpenCL-Headers
    cudadrv
  COMPILE_DEFINITIONS
    $<$<PLATFORM_ID:Windows>:__SYCL_BUILD_SYCL_DLL>
  # Filter symbols based on the scope defined in the script file,
  # and export pi* function symbols in the library.
  VERSION_SCRIPT "${linker_script}"
  HIDE_SYMBOLS
)

add_dependencies(sycl-toolchain pi_cuda)

install(TARGETS pi_cuda
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_cuda
  RUNTIME DESTINATION "bin" COMPONENT pi_cuda
)
add_llvm_install_targets(install-pi_cuda
  COMPONENT pi_cuda
  DEPENDS pi_cuda
  )
