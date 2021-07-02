add_sycl_unittest(KernelAndProgramTests OBJECT
  KernelRelease.cpp
  KernelInfo.cpp
  DeviceInfo.cpp
  PersistentDeviceCodeCache.cpp
)
add_subdirectory(device)
