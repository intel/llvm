set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(PipeTests OBJECT
  host_pipe_registration.cpp
)

add_dependencies(PipeTests sycl)
target_include_directories(PipeTests PRIVATE SYSTEM ${sycl_inc_dir})
