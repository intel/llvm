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

add_library(pi_opencl SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "pi_opencl.cpp"
  )

add_dependencies(sycl-toolchain pi_opencl)

set_target_properties(pi_opencl PROPERTIES LINKER_LANGUAGE CXX)

#preprocessor definitions for compiling a target's sources. We do not need it for pi_opencl
target_include_directories(pi_opencl PRIVATE "${sycl_inc_dir}")

#link pi_opencl with OpenCL headers and ICD Loader.
target_link_libraries(pi_opencl
  PRIVATE
    OpenCL-Headers
    OpenCL-ICD
)

add_common_options(pi_opencl)

install(TARGETS pi_opencl
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_opencl
  RUNTIME DESTINATION "bin" COMPONENT pi_opencl)
