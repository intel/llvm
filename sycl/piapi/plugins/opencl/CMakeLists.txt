#TODO:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)

find_package(OpenCL REQUIRED)

# Plugin for OpenCL
# Create Shared library for libpi_opencl.so.
#TODO: remove dependency on pi.hpp in sycl project.
#TODO: Currently, the pi.hpp header is common between sycl and plugin library sources. 
#This can be changed by copying the pi.hpp file in the plugins project.

add_library(pi_opencl SHARED
  "${pi_include_dir}/pi/pi.h"
  "pi_opencl.cpp"
)
set_target_properties(pi_opencl PROPERTIES
  ARCHIVE_OUTPUT_DIRECTORY "${pi_library_output_dir}"
  LIBRARY_OUTPUT_DIRECTORY "${pi_library_output_dir}"
  RUNTIME_OUTPUT_DIRECTORY "${pi_binary_output_dir}"
)

target_include_directories(pi_opencl PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(pi_opencl PUBLIC piapi OpenCL::OpenCL)
target_link_libraries(pi_opencl PRIVATE pi_export_library)

install(TARGETS pi_opencl
  EXPORT piapiTargets
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_opencl
  RUNTIME DESTINATION "bin" COMPONENT pi_opencl
)
export(TARGETS pi_opencl APPEND
  FILE "${PROJECT_BINARY_DIR}/piapiTargets.cmake"
)
