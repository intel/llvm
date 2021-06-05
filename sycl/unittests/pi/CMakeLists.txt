set(CMAKE_CXX_EXTENSIONS OFF)

# Enable exception handling for these unit tests
set(LLVM_REQUIRES_EH 1)
add_sycl_unittest(PiTests OBJECT
  EnqueueMemTest.cpp
  PiMock.cpp
  PlatformTest.cpp
)

add_dependencies(PiTests sycl)
target_include_directories(PiTests PRIVATE SYSTEM ${sycl_inc_dir})

if(SYCL_BUILD_PI_CUDA)
    add_subdirectory(cuda)
endif()
