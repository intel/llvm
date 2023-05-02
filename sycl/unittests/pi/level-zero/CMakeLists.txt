add_sycl_unittest(PiLevelZeroTests OBJECT
  test_ext_command_buffers.cpp
)

add_dependencies(PiLevelZeroTests sycl)

target_compile_definitions(PiLevelZeroTests
  PRIVATE
  GTEST_HAS_COMBINE=1)

target_include_directories(PiLevelZeroTests
  PRIVATE
  "../"
  "${sycl_inc_dir}/sycl/detail/"
  "${sycl_inc_dir}"
  "${sycl_plugin_dir}/level_zero/"
)

message("LevelZero var: ${LEVEL_ZERO_LOADER}")

if(TARGET level-zero-loader)
  add_dependencies(PiLevelZeroTests level-zero-loader)
endif()
