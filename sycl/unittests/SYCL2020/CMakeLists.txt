set(CMAKE_CXX_EXTENSIONS OFF)

# Enable exception handling for these unit tests
set(LLVM_REQUIRES_EH 1)
add_sycl_unittest(SYCL2020Tests OBJECT
  ExtDefaultContext.cpp
  GetNativeOpenCL.cpp
  SpecConstDefaultValues.cpp
  KernelBundle.cpp
)

