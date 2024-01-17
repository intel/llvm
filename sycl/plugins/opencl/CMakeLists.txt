# Plugin for OpenCL
# Create Shared library for libpi_opencl.so.

# Get the OpenCL adapter sources so they can be shared with the OpenCL PI plugin
get_target_property(UR_OPENCL_ADAPTER_SOURCES ur_adapter_opencl SOURCES)

add_sycl_plugin(opencl
  SOURCES
    ${UR_OPENCL_ADAPTER_SOURCES}
    # Some code is shared with the UR adapter
    "../unified_runtime/pi2ur.hpp"
    "${sycl_inc_dir}/sycl/detail/pi.h"
    "${sycl_inc_dir}/sycl/detail/pi.hpp"
    "pi_opencl.cpp"
    "pi_opencl.hpp"
  INCLUDE_DIRS
    ${sycl_inc_dir}
    ${CMAKE_CURRENT_SOURCE_DIR}/../unified_runtime # for Unified Runtime
    ${UNIFIED_RUNTIME_SOURCE_DIR}/source/ # for adapters/opencl
  LIBRARIES
    UnifiedRuntime-Headers
    UnifiedRuntimeCommon
    OpenCL-ICD
)

if (WIN32)
  target_compile_definitions(pi_opencl PRIVATE
    WIN32_LEAN_AND_MEAN
    NOMINMAX
  )
endif()

set_target_properties(pi_opencl PROPERTIES LINKER_LANGUAGE CXX)
