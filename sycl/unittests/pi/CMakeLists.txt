set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_llvm_unittest(PiTests
  PlatformTest.cpp
)

add_dependencies(PiTests sycl)
target_link_libraries(PiTests PRIVATE sycl LLVMTestingSupport)

if(NOT MSVC)
  target_compile_options(PiTests PRIVATE -fexceptions)
endif()
