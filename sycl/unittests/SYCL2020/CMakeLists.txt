set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(SYCL2020Tests OBJECT
  GetNativeOpenCL.cpp
  SpecializationConstant.cpp
  KernelBundle.cpp
  KernelID.cpp
)

