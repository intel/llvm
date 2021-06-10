
# PI Esimd CPU  library
# Create Shared library for libpi_esimd_cpu.so.

include_directories("${sycl_inc_dir}")
include_directories(${OpenCL_INCLUDE_DIR})
include_directories(${LIBCMRT_INCLUDE})

add_library(pi_esimd_cpu SHARED
  "${sycl_inc_dir}/CL/sycl/detail/pi.h"
  "pi_esimd_cpu.cpp"
)

if (MSVC)
  # by defining __SYCL_BUILD_SYCL_DLL, we can use __declspec(dllexport)
  # which are individually tagged for all pi* symbols in pi.h
  target_compile_definitions(pi_esimd_cpu PRIVATE __SYCL_BUILD_SYCL_DLL)
else()
  # we set the visibility of all symbols 'hidden' by default.
  # In pi.h file, we set exported symbols with visibility==default individually
  target_compile_options(pi_esimd_cpu PUBLIC -fvisibility=hidden)

  # This script file is used to allow exporting pi* symbols only.
  # All other symbols are regarded as local (hidden)
  set(linker_script "${CMAKE_CURRENT_SOURCE_DIR}/../ld-version-script.txt")

  # Filter symbols based on the scope defined in the script file,
  # and export pi* function symbols in the library.
  target_link_libraries( pi_esimd_cpu
    PRIVATE "-Wl,--version-script=${linker_script}"
  )
endif()

add_dependencies(sycl-toolchain pi_esimd_cpu)

add_dependencies(pi_esimd_cpu
  OpenCL-Headers)

target_link_libraries(pi_esimd_cpu PRIVATE sycl)
set_target_properties(pi_esimd_cpu PROPERTIES LINKER_LANGUAGE CXX)

add_common_options(pi_esimd_cpu)

install(TARGETS pi_esimd_cpu
    LIBRARY DESTINATION "lib${LLVM_LIBDIR_SUFFIX}" COMPONENT pi_esimd_cpu
    RUNTIME DESTINATION "bin" COMPONENT pi_esimd_cpu)
