// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %{build} %level_zero_options -o %t.ze.out
// RUN: %{run} %t.ze.out

#include <level_zero/ze_api.h>

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>

constexpr auto BE = sycl::backend::ext_oneapi_level_zero;

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

  ze_kernel_handle_t Handle = sycl::get_native<BE>(Kernel);

  ze_kernel_properties_t KernelProperties = {
      ZE_STRUCTURE_TYPE_KERNEL_PROPERTIES, 0};
  ze_result_t Err = zeKernelGetProperties(Handle, &KernelProperties);
  assert(Err == ZE_RESULT_SUCCESS);

  // SYCL2020 4.5.1.2 - verify exception errc
  try {
    // this test is L0 only, so we ask for an unavailable backend.
    auto BE2 = sycl::get_native<sycl::backend::opencl>(Q);
    assert(false && "we should not be here.");
  } catch (sycl::exception e) {
    assert(e.code() == sycl::errc::backend_mismatch && "wrong error code");
  }

  return 0;
}
