set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(PiTests OBJECT
  EnqueueMemTest.cpp
  PiMock.cpp
  PlatformTest.cpp
  pi_arguments_handler.cpp
  piInteropRetain.cpp
)

add_dependencies(PiTests sycl)
target_include_directories(PiTests PRIVATE SYSTEM ${sycl_inc_dir})
target_include_directories(PiTests PRIVATE ${sycl_src_dir}/../tools/xpti_helpers)

if(SYCL_BUILD_PI_CUDA)
    add_subdirectory(cuda)
endif()
