#TODO:
#1. Figure out why CMP0057 has to be set. Should have been taken care of earlier in the build
#2. Use AddLLVM to modify the build and access config options
#cmake_policy(SET CMP0057 NEW)
#include(AddLLVM)

# Plugin for OpenCL
# Create Shared library for libpi_opencl.so.
#TODO: remove dependency on pi.h in sycl project.
#TODO: Currently, the pi.h header is common between sycl and plugin library sources.
#This can be changed by copying the pi.h file in the plugins project.

find_package(Threads REQUIRED)

add_sycl_plugin(opencl
  SOURCES
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "pi_opencl.cpp"
  LIBRARIES
    OpenCL-ICD
    Threads::Threads
)

set_target_properties(pi_opencl PROPERTIES LINKER_LANGUAGE CXX)
