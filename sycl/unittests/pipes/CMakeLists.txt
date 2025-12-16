set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(PipeTests OBJECT
  host_pipe_registration.cpp
)

add_dependencies(PipeTests_Non_Preview_Tests sycl)
target_include_directories(PipeTests_Non_Preview_Tests PRIVATE SYSTEM ${sycl_inc_dir})
add_dependencies(PipeTests_Preview_Tests sycl)
target_include_directories(PipeTests_Preview_Tests PRIVATE SYSTEM ${sycl_inc_dir})
