set(xpti_include_dir "")
if(SYCL_ENABLE_XPTI_TRACING)
  set(xpti_include_dir "${LLVM_EXTERNAL_XPTI_SOURCE_DIR}/include")
endif()

# Set values for the library
set(PI_BUILD_CUDA "${SYCL_BUILD_PI_CUDA}")
set(PI_BUILD_LEVEL_ZERO ON)
set(SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE "${SYCL_EP_LEVEL_ZERO_LOADER_SKIP_AUTO_UPDATE}")
set(PI_LIBDIR_SUFFIX "${LLVM_LIBDIR_SUFFIX}")
set(PI_XPTI_INCLUDE_DIR "${xpti_include_dir}")

# Include the library
add_subdirectory(piapi)

# Enables the piapi library
function(add_piapi_library)
  add_library(piapi::piapi ALIAS piapi)
endfunction()

# PI has special code for DPC++ integration
target_compile_definitions(piapi INTERFACE PI_DPCPP_INTEGRATION)
target_include_directories(piapi
  INTERFACE
    "${sycl_inc_dir}"
)

if (MSVC AND CMAKE_BUILD_TYPE MATCHES "Debug")
  set(XPTI_LIB xptid)
else()
  set(XPTI_LIB xpti)
endif()
if(SYCL_ENABLE_XPTI_TRACING)
  #target_link_libraries(pi_level_zero PRIVATE ${XPTI_LIB})
endif()

