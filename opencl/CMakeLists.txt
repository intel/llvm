cmake_minimum_required(VERSION 3.20.0)

include(FetchContent)

# Suppress warnings related to dependencies
if (MSVC)
  add_definitions(
    -wd4133 # Suppress ''function': incompatible types - from 'char [9]' to 'LPCWSTR''
    -wd4152 # Suppress 'nonstandard extension, function/data pointer conversion in expression'
    -wd4201 # Suppress 'nonstandard extension used: nameless struct/union'
  )
endif()

# LLVM build sets this OFF by default, but we need OpenCL to be built as shared
# library.
set(BUILD_SHARED_LIBS ON)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/../unified-runtime/cmake")

include(FetchOpenCL)

add_subdirectory(opencl-aot)
