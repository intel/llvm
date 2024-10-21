# Utility project providing various functionalities for SPIR-V devices
# without native support of these functionalities.

include(${CMAKE_CURRENT_SOURCE_DIR}/../libclc/cmake/modules/AddLibclc.cmake)

set(CMAKE_MODULE_PATH
  ${CMAKE_MODULE_PATH}
  "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules"
  )

if(EXISTS ${FETCHCONTENT_BASE_DIR}/unified-runtime-src)
  set(UR_SANITIZER_INCLUDE_DIR
      ${FETCHCONTENT_BASE_DIR}/unified-runtime-src/source/loader/layers/sanitizer)
elseif(EXISTS ${SYCL_UR_SOURCE_DIR})
  set(UR_SANITIZER_INCLUDE_DIR
      ${SYCL_UR_SOURCE_DIR}/source/loader/layers/sanitizer)
else()
  message(WARNING "Unified Runtime source directory is not provided, \
                   will not build libdevice sanitizer")
endif()

# Build libdevice for SYCL.
include(SYCLLibdevice)

set(LIBDEVICE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(LIBDEVICE_LIBRARY_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})
set(LIBDEVICE_RUNTIME_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
set(LIBDEVICE_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})

add_subdirectory(test)
