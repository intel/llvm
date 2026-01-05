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
  KernelArgs.cpp
  KernelInfoShortcuts.cpp
)
target_compile_definitions(KernelAndProgramTests_Non_Preview_Tests PRIVATE __SYCL_INTERNAL_API)
target_compile_definitions(KernelAndProgramTests_Preview_Tests PRIVATE __SYCL_INTERNAL_API __INTEL_PREVIEW_BREAKING_CHANGES)
