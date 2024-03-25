#define SYCL2020_DISABLE_DEPRECATION_WARNINGS

#include <sycl/detail/core.hpp>

class KernelName;

int main() {
  sycl::queue Queue;

  sycl::device Dev = Queue.get_device();

  sycl::context Ctx = Queue.get_context();

  sycl::kernel_bundle KernelBundle =
      sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

  if (0)
    Queue.submit([&](sycl::handler &CGH) {
      CGH.single_task<KernelName>([=]() {}); // Actual kernel does not matter
    });

  try {
    auto ExecBundle = sycl::build(KernelBundle);
  } catch (...) {
    // Ignore all exceptions
  }

  try {
    auto KernelBundleObject =
        sycl::compile(KernelBundle, KernelBundle.get_devices());

    auto KernelBundleExecutable =
        sycl::link({KernelBundleObject}, KernelBundleObject.get_devices());
  } catch (...) {
    // Ignore all exceptions
  }

  return 0;
}
