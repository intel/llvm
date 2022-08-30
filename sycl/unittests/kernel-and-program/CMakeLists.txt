add_sycl_unittest(KernelAndProgramTests OBJECT
  MultipleDevsCache.cpp
  DeviceInfo.cpp
  PersistentDeviceCodeCache.cpp
)
target_compile_definitions(KernelAndProgramTests PRIVATE -D__SYCL_INTERNAL_API)
