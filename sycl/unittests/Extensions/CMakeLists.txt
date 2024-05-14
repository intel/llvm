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
)

add_subdirectory(CommandGraph)
