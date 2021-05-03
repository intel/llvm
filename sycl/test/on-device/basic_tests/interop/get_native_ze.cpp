// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.ze.out
// RUN: %t.ze.out

#include <level_zero/ze_api.h>

#include <CL/sycl/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

constexpr auto BE = sycl::backend::level_zero;

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

  ze_kernel_handle_t Handle = Kernel.get_native<BE>();

  ze_kernel_properties_t KernelProperties;
  ze_result_t Err = zeKernelGetProperties(Handle, &KernelProperties);
  assert(Err == ZE_RESULT_SUCCESS);

  return 0;
}
