add_library(sycl_unittest_helpers STATIC
  PiMock.cpp
)

target_include_directories(sycl_unittest_helpers PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_include_directories(sycl_unittest_helpers PRIVATE
  ${sycl_src_dir}
  ${sycl_inc_dir}
  )

