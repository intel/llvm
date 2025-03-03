get_target_property(SYCL_BINARY_DIR sycl-toolchain BINARY_DIR)

add_library(mockOpenCL SHARED EXCLUDE_FROM_ALL mock_opencl.cpp)
set_target_properties(mockOpenCL PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${SYCL_BINARY_DIR}/unittests/lib
  RUNTIME_OUTPUT_DIRECTORY ${SYCL_BINARY_DIR}/unittests/lib
  ARCHIVE_OUTPUT_DIRECTORY ${SYCL_BINARY_DIR}/unittests/lib
  OUTPUT_NAME OpenCL)
