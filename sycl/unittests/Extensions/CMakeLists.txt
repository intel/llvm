set(CMAKE_CXX_EXTENSIONS OFF)

add_sycl_unittest(ExtensionsTests OBJECT
  CurrentDevice.cpp
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
  ProfilingTag.cpp
  KernelProperties.cpp
  NoDeviceIPVersion.cpp
  WorkGroupMemoryBackendArgument.cpp
  GetLastEvent.cpp
  BFloat16.cpp
  LaunchQueries.cpp
  EventMode.cpp
  DeviceInfo.cpp
  RootGroup.cpp
)

add_subdirectory(CommandGraph)
add_subdirectory(VirtualFunctions)
add_subdirectory(VirtualMemory)
add_subdirectory(NumComputeUnits)
add_subdirectory(FreeFunctionCommands)
add_subdirectory(KernelQueries)
