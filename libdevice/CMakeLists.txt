# Utility project providing various functionalities for SPIR-V devices
# without native support of these functionalities.

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

# Build libdevice for SYCL.
include(SYCLLibdevice)

set(LIBDEVICE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBDEVICE_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(LIBDEVICE_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(LIBDEVICE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(test)
