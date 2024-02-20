add_sycl_unittest(KernelAndProgramTests OBJECT
  Cache.cpp
  MultipleDevsCache.cpp
  KernelRelease.cpp
  KernelInfo.cpp
  DeviceInfo.cpp
  PersistentDeviceCodeCache.cpp
  KernelBuildOptions.cpp
  OutOfResources.cpp
)
target_compile_definitions(KernelAndProgramTests PRIVATE -D__SYCL_INTERNAL_API)
