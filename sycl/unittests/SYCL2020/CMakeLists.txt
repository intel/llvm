set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest_both(SYCL2020Tests SYCL2020TestsPreview OBJECT
  GetNativeOpenCL.cpp
  SpecializationConstant.cpp
  KernelBundle.cpp
  KernelBundleStateFiltering.cpp
  KernelID.cpp
  HasExtension.cpp
  IsCompatible.cpp
  DeviceGetInfoAspects.cpp
  DeviceAspectTraits.cpp
  AtomicMemoryOrderCapabilities.cpp
  AtomicMemoryScopeCapabilities.cpp
  AtomicFenceCapabilities.cpp
)

