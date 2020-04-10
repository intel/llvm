set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO: move each tool in its own sub-directory

add_executable(lspi lspi.cpp)
add_dependencies(lspi sycl)
target_include_directories(lspi PRIVATE "${sycl_inc_dir}")
target_link_libraries(lspi
    PRIVATE sycl
    PRIVATE OpenCL::Headers
    PRIVATE ${OpenCL_LIBRARIES})

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
