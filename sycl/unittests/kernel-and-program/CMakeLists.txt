add_sycl_unittest_both(KernelAndProgramTests KernelAndProgramTestsPreview OBJECT
  Cache.cpp
  MultipleDevsCache.cpp
  KernelRelease.cpp
  KernelInfo.cpp
  DeviceInfo.cpp
  PersistentDeviceCodeCache.cpp
  KernelBuildOptions.cpp
  OutOfResources.cpp
  InMemCacheEviction.cpp
)
target_compile_definitions(KernelAndProgramTests PRIVATE -D__SYCL_INTERNAL_API)
target_compile_definitions(KernelAndProgramTestsPreview PRIVATE -D__SYCL_INTERNAL_API)
