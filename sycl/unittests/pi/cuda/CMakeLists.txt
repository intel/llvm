set(LLVM_REQUIRES_EH 1)
add_sycl_unittest(PiCudaTests OBJECT
  test_base_objects.cpp
  test_commands.cpp
  test_device.cpp
  test_interop_get_native.cpp
  test_kernels.cpp
  test_mem_obj.cpp
  test_primary_context.cpp
  test_queue.cpp
  test_events.cpp
)

add_dependencies(PiCudaTests sycl)

target_link_libraries(PiCudaTests PRIVATE
  LLVMTestingSupport
  OpenCL-Headers)

target_include_directories(
  PiCudaTests PUBLIC 
  ${CUDA_INCLUDE_DIRS} 
  "../"
  "${sycl_inc_dir}/CL/sycl/detail/"
  ${sycl_inc_dir}
  "${sycl_plugin_dir}/cuda/"
)
