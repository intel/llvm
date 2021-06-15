add_sycl_unittest(KernelAndProgramTests OBJECT
  KernelRelease.cpp
  KernelInfo.cpp
  PersistentDeviceCodeCache.cpp
)
add_subdirectory(device)
