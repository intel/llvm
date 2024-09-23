// REQUIRES: opencl, opencl_dev_kit
// RUN: %{build} %opencl_options -o %t.ocl.out
// RUN: %{run} %t.out

#include <CL/cl.h>

#include <sycl/backend.hpp>
#include <sycl/backend/opencl.hpp>
#include <sycl/detail/core.hpp>

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

  cl_kernel Handle = sycl::get_native<BE>(Kernel);

  size_t Size = 0;
  cl_int Err =
      clGetKernelInfo(Handle, CL_KERNEL_FUNCTION_NAME, 0, nullptr, &Size);
  assert(Err == CL_SUCCESS);

  std::vector<cl_program> Progs = sycl::get_native<BE>(KernelBundle);
  for (cl_program Prog : Progs) {
    Err = clGetProgramInfo(Prog, CL_PROGRAM_REFERENCE_COUNT, 0, nullptr, &Size);
    assert(Err == CL_SUCCESS);
  }

  return 0;
}
