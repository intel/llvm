set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Enable exception handling for these unit tests
set(LLVM_REQUIRES_EH 1)
add_llvm_unittest(PiTests
  PlatformTest.cpp
)

add_dependencies(PiTests sycl)
target_link_libraries(PiTests PRIVATE sycl LLVMTestingSupport)