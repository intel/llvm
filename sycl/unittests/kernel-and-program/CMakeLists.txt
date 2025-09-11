add_sycl_unittest(KernelAndProgramTests OBJECT
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
target_compile_definitions(KernelAndProgramTests_non_preview PRIVATE -D__SYCL_INTERNAL_API)
target_compile_definitions(KernelAndProgramTests_preview PRIVATE -D__SYCL_INTERNAL_API)
