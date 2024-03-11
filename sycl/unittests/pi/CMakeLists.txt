set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(PiTests OBJECT
  PiMock.cpp
  PiUtility.cpp
  pi_arguments_handler.cpp
  piInteropRetain.cpp
)

add_dependencies(PiTests sycl)
target_include_directories(PiTests PRIVATE SYSTEM ${sycl_inc_dir})
target_include_directories(PiTests PRIVATE ${sycl_src_dir}/../tools/xpti_helpers)

