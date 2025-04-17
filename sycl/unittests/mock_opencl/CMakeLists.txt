get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

# Linux looks up ur_adapter_opencl rather than libOpenCL.
# On Windows, this is copied into libOpenCL.dll.
if(WIN32)
  set(LIBNAME OpenCL)
else()
  set(LIBNAME ur_adapter_opencl)
endif()

add_library(mockOpenCL SHARED EXCLUDE_FROM_ALL mock_opencl.cpp)
set_target_properties(mockOpenCL PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${SYCL_BINARY_DIR}/unittests/lib
  LIBRARY_OUTPUT_NAME ${LIBNAME}
  RUNTIME_OUTPUT_DIRECTORY ${SYCL_BINARY_DIR}/unittests/lib
  RUNTIME_OUTPUT_NAME ${LIBNAME}
)
