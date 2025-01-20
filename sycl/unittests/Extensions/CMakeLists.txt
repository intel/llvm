set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(ExtensionsTests OBJECT
  DefaultContext.cpp
  FPGADeviceSelectors.cpp
  DeviceArchitecture.cpp
  USMMemcpy2D.cpp
  DeviceGlobal.cpp
  OneAPISubGroupMask.cpp
  USMP2P.cpp
  CompositeDevice.cpp
  OneAPIProd.cpp
  EnqueueFunctionsEvents.cpp
  DiscardEvent.cpp
  ProfilingTag.cpp
  KernelProperties.cpp
  NoDeviceIPVersion.cpp
  WorkGroupMemoryBackendArgument.cpp
  GetLastEvent.cpp
  BFloat16.cpp
  EventMode.cpp
)

add_subdirectory(CommandGraph)
add_subdirectory(VirtualFunctions)
add_subdirectory(VirtualMemory)
