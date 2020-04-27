set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(sycl-ls sycl-ls.cpp)
add_dependencies(sycl-ls sycl)
target_include_directories(sycl-ls PRIVATE "${sycl_inc_dir}")
target_link_libraries(sycl-ls
  PRIVATE
    sycl
    OpenCL::Headers
    ${OpenCL_LIBRARIES}
)
