# Utility project providing various functionalities for SPIR-V devices
# without native support of these functionalities.

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

# Build libdevice for SYCL.
include(SYCLLibdevice)
