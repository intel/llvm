set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(PipeTests OBJECT
  host_pipe_registration.cpp
)

add_dependencies(PipeTests_non_preview sycl)
target_include_directories(PipeTests_non_preview PRIVATE SYSTEM ${sycl_inc_dir})
add_dependencies(PipeTests_preview sycl)
target_include_directories(PipeTests_preview PRIVATE SYSTEM ${sycl_inc_dir})
