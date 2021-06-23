// REQUIRES: opencl, opencl_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %opencl_options %s -o %t.ocl.out
// RUN: %t.ocl.out

#include <CL/cl.h>

#include <CL/sycl/backend/opencl.hpp>
#include <sycl/sycl.hpp>

constexpr auto BE = sycl::backend::opencl;

class TestKernel;

int main() {
  sycl::queue Q;

  if (0) {
    Q.submit([](sycl::handler &CGH) { CGH.single_task<TestKernel>([] {}); });
  }

  sycl::kernel_id KernelID = sycl::get_kernel_id<TestKernel>();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::executable>(Q.get_context());

  sycl::kernel Kernel = KernelBundle.get_kernel(KernelID);

  cl_kernel Handle = Kernel.get_native<BE>();

  size_t Size = 0;
  cl_int Err =
      clGetKernelInfo(Handle, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &Size);
  assert(Err == CL_SUCCESS);

  return 0;
}
