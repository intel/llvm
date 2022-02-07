add_library(sycl_unittest_helpers OBJECT EXCLUDE_FROM_ALL
  PiMock.cpp
)

target_include_directories(sycl_unittest_helpers PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(sycl_unittest_helpers PRIVATE
  ${sycl_src_dir}
  ${sycl_inc_dir}
  )

target_link_libraries(sycl_unittest_helpers
  PRIVATE OpenCL-Headers
)
