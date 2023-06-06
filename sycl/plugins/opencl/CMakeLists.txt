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

add_sycl_plugin(opencl
  SOURCES
    # Some code is shared with the UR adapter
    "../unified_runtime/pi2ur.hpp"
    "../unified_runtime/pi2ur.cpp"
    "../unified_runtime/ur/ur.hpp"
    "../unified_runtime/ur/ur.cpp"
    "../unified_runtime/ur/usm_allocator.cpp"
    "../unified_runtime/ur/usm_allocator.hpp"
    "../unified_runtime/ur/adapters/opencl/common.cpp"
    "../unified_runtime/ur/adapters/opencl/common.hpp"
    "../unified_runtime/ur/adapters/opencl/context.cpp"
    "../unified_runtime/ur/adapters/opencl/context.hpp"
    "../unified_runtime/ur/adapters/opencl/device.cpp"
    "../unified_runtime/ur/adapters/opencl/device.hpp"
    "../unified_runtime/ur/adapters/opencl/enqueue.cpp"
    "../unified_runtime/ur/adapters/opencl/kernel.cpp"
    "../unified_runtime/ur/adapters/opencl/platform.cpp"
    "../unified_runtime/ur/adapters/opencl/platform.hpp"
    "../unified_runtime/ur/adapters/opencl/sampler.cpp"
    "../unified_runtime/ur/adapters/opencl/memory.cpp"
    "../unified_runtime/ur/adapters/opencl/usm.cpp"
    "../unified_runtime/ur/adapters/opencl/program.cpp"
    "../unified_runtime/ur/adapters/opencl/event.cpp"
    "../unified_runtime/ur/adapters/opencl/queue.cpp"
    # ---
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "pi_opencl.cpp"
  INCLUDE_DIRS
    ${sycl_inc_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime
  LIBRARIES
    UnifiedRuntime-Headers
    OpenCL-ICD
)

set_target_properties(pi_opencl PROPERTIES LINKER_LANGUAGE CXX)
