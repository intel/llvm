set(CMAKE_CXX_EXTENSIONS OFF)

# Enable exception handling for these unit tests
set(LLVM_REQUIRES_EH 1)
add_sycl_unittest(PiTests OBJECT
  EnqueueMemTest.cpp
  PiMock.cpp
  PlatformTest.cpp
  EventTest.cpp
)

add_dependencies(PiTests sycl)
target_link_libraries(PiTests PRIVATE sycl LLVMTestingSupport OpenCL-Headers)
target_include_directories(PiTests PRIVATE SYSTEM ${sycl_inc_dir})

if(SYCL_BUILD_PI_CUDA)
    add_subdirectory(cuda)
endif()
