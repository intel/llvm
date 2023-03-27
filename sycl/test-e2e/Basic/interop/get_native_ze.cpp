// REQUIRES: level_zero, level_zero_dev_kit
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %level_zero_options %s -o %t.ze.out
// RUN: %t.ze.out

// Temporarily disable on L0 due to fails in CI
// UNSUPPORTED: level_zero

#include <level_zero/ze_api.h>

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <sycl/sycl.hpp>

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
