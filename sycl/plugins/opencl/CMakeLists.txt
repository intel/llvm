# Plugin for OpenCL
# Create Shared library for libpi_opencl.so.
#TODO: remove dependency on pi.h in sycl project.
#TODO: Currently, the pi.h header is common between sycl and plugin library sources.
#This can be changed by copying the pi.h file in the plugins project.

add_sycl_library(pi_opencl SHARED
  SOURCES
    "${sycl_inc_dir}/CL/sycl/detail/pi.h"
    "pi_opencl.cpp"
  INCLUDES
    ${sycl_inc_dir}
  COMPILE_DEFINITIONS
    $<$<PLATFORM_ID:Windows>:__SYCL_BUILD_SYCL_DLL>
  LINK
    OpenCL-Headers
    OpenCL-ICD
  # This script file is used to allow exporting pi* symbols only.
  # All other symbols are regarded as local (hidden)
  VERSION_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt"
  # we set the visibility of all symbols 'hidden' by default.
  # In pi.h file, we set exported symbols with visibility==default individually
  HIDE_SYMBOLS
)

add_dependencies(sycl-toolchain pi_opencl)

install(TARGETS pi_opencl
  LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_opencl
  RUNTIME DESTINATION "bin" COMPONENT pi_opencl)
add_llvm_install_targets(install-pi_opencl
  COMPONENT pi_opencl
  DEPENDS pi_opencl
  )
