add_sycl_unittest(KernelAndProgramTests OBJECT
  Cache.cpp
  MultipleDevsCache.cpp
  KernelBuildOptions.cpp
  KernelRelease.cpp
  KernelInfo.cpp
  DeviceInfo.cpp
  PersistentDeviceCodeCache.cpp
)
target_compile_definitions(KernelAndProgramTests PRIVATE -D__SYCL_INTERNAL_API)
