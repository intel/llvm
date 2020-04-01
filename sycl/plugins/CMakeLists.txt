if(SYCL_BUILD_PI_CUDA)
 add_subdirectory(cuda)
endif()

add_subdirectory(opencl)
