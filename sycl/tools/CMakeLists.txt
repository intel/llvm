set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_subdirectory(sycl-ls)

# TODO: move each tool in its own sub-directory
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
