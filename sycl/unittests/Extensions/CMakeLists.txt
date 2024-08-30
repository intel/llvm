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
)

add_subdirectory(CommandGraph)

# Disabled pending UR fix adding setDataAs to mock dummy handle.
#add_subdirectory(VirtualFunctions)
