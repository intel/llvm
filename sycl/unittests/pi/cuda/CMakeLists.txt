set(LLVM_REQUIRES_EH 1)
add_sycl_unittest(PiCudaTests
  test_base_objects.cpp
  test_commands.cpp
  test_device.cpp
  test_kernels.cpp
  test_mem_obj.cpp
  test_queue.cpp
  test_events.cpp
)

add_dependencies(PiCudaTests sycl)

target_link_libraries(PiCudaTests PRIVATE
  sycl
  LLVMTestingSupport
  OpenCL-Headers)

target_include_directories(
  PiCudaTests PUBLIC 
  ${CUDA_INCLUDE_DIRS} 
  "${sycl_inc_dir}/CL/sycl/detail/"
  ${sycl_inc_dir}
  "${sycl_plugin_dir}/cuda/"
)
