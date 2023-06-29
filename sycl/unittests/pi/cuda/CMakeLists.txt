add_sycl_unittest(PiCudaTests OBJECT
  test_base_objects.cpp
  test_commands.cpp
  test_contexts.cpp
  test_device.cpp
  test_interop_get_native.cpp
  test_kernels.cpp
  test_mem_obj.cpp
  test_primary_context.cpp
  test_sampler_properties.cpp
)

add_dependencies(PiCudaTests sycl)

target_compile_definitions(PiCudaTests
  PRIVATE
    GTEST_HAS_COMBINE=1)

target_include_directories(PiCudaTests
  PRIVATE
    "../"
    "${sycl_inc_dir}/sycl/detail/"
    "${sycl_inc_dir}"
    "${sycl_plugin_dir}/cuda/"
    "${sycl_plugin_dir}/unified_runtime/"
)

target_link_libraries(PiCudaTests
  PRIVATE
    cudadrv
    UnifiedRuntime-Headers
)
