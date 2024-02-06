add_sycl_unittest(PiHipTests OBJECT
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

add_dependencies(PiHipTests sycl)

target_compile_definitions(PiHipTests
  PRIVATE
    GTEST_HAS_COMBINE=1)

target_include_directories(PiHipTests
  PRIVATE
    "../"
    "${sycl_inc_dir}/sycl/detail/"
    "${sycl_inc_dir}"
    "${sycl_plugin_dir}/hip/"
    "${sycl_plugin_dir}/unified_runtime/"
)

if("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "AMD")
  # Set HIP define to select AMD platform
  target_compile_definitions(PiHipTests PRIVATE __HIP_PLATFORM_AMD__)
elseif("${SYCL_BUILD_PI_HIP_PLATFORM}" STREQUAL "NVIDIA")
  # Set HIP define to select NVIDIA platform
  target_compile_definitions(PiHipTests PRIVATE __HIP_PLATFORM_NVIDIA__)
else()
  message(FATAL_ERROR "Unspecified PI HIP platform, please set SYCL_BUILD_PI_HIP_PLATFORM to 'AMD' or 'NVIDIA'")
endif()

target_link_libraries(PiHipTests
  PRIVATE
    rocmdrv
    UnifiedRuntime-Headers
)
