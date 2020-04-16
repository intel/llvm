set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(get_device_count_by_type get_device_count_by_type.cpp)
add_dependencies(get_device_count_by_type ocl-headers ocl-icd)

target_link_libraries(get_device_count_by_type
  PRIVATE
    OpenCL::Headers
    ${OpenCL_LIBRARIES}
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:cudadrv>
)
target_compile_definitions(get_device_count_by_type
  PRIVATE
    $<$<BOOL:${SYCL_BUILD_PI_CUDA}>:USE_PI_CUDA>
)

add_executable(sycl-check sycl-check.cpp)
add_dependencies(sycl-check sycl)
target_include_directories(sycl-check PRIVATE "${sycl_inc_dir}")
target_link_libraries(sycl-check
  PRIVATE
    sycl
    OpenCL::Headers
    ${OpenCL_LIBRARIES})

#Minimum supported version of Intel's OCL GPU and CPU devices
target_compile_definitions(sycl-check
  PRIVATE
    MIN_INTEL_OCL_GPU_VERSION=\"18.47.11882\"
    MIN_INTEL_OCL_CPU_VERSION=\"18.1.0.0901\",\"7.6.0.1202\"
)
