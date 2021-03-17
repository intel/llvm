// RUN: %clangxx -fsycl -fsycl-device-code-split=per_kernel -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %t.out
//
// -fsycl-device-code-split is not supported for cuda
// UNSUPPORTED: cuda

#include <CL/sycl.hpp>

#include <vector>

class Kernel1Name;
class Kernel2Name;

int main() {
  sycl::queue Q;

  // No support for host device so far.
  if (Q.is_host())
    return 0;

  const sycl::context Ctx = Q.get_context();
  const sycl::device Dev = Q.get_device();

  Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel1Name>([]() {}); });
  Q.submit([](sycl::handler &CGH) { CGH.single_task<Kernel2Name>([]() {}); });

  sycl::kernel_id Kernel1ID = sycl::get_kernel_id<Kernel1Name>();
  sycl::kernel_id Kernel2ID = sycl::get_kernel_id<Kernel2Name>();

  {
    sycl::kernel_bundle KernelBundle1 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev});

    sycl::kernel_bundle KernelBundleCopy = KernelBundle1;
    assert(KernelBundleCopy == KernelBundle1);
    assert(!(KernelBundleCopy != KernelBundle1));
    assert(false == KernelBundle1.empty());
    assert(Ctx.get_platform().get_backend() == KernelBundle1.get_backend());
    assert(KernelBundle1.get_context() == Ctx);
    assert(KernelBundle1.get_devices() == (std::vector<sycl::device>){Dev});
    assert(KernelBundle1.has_kernel(Kernel1ID));
    assert(KernelBundle1.has_kernel(Kernel2ID));
    assert(KernelBundle1.has_kernel(Kernel1ID, Dev));
    assert(KernelBundle1.has_kernel(Kernel2ID, Dev));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel1ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel1ID);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel2ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel1ID, &Dev](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel1ID, Dev);
        }));

    assert(std::any_of(
        KernelBundle1.begin(), KernelBundle1.end(),
        [&Kernel2ID, &Dev](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID, Dev);
        }));
  }

  // The following check relies on "-fsycl-device-code-split=per_kernel" option,
  // so it is expected that each kernel in a separate device image.
  // Verify that get_kernel_bundle filters out device images based on vector
  // of kernel_id's and Selector.

  {
    sycl::kernel_bundle KernelBundle2 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                           {Kernel1ID});
    assert(KernelBundle2.has_kernel(Kernel1ID));
    assert(!KernelBundle2.has_kernel(Kernel2ID));

    auto Selector =
        [&Kernel2ID](
            const sycl::device_image<sycl::bundle_state::input> &DevImage) {
          return DevImage.has_kernel(Kernel2ID);
        };

    sycl::kernel_bundle KernelBundle3 =
        sycl::get_kernel_bundle<sycl::bundle_state::input>(Ctx, {Dev},
                                                           Selector);
    assert(!KernelBundle3.has_kernel(Kernel1ID));
    assert(KernelBundle3.has_kernel(Kernel2ID));
  }

  return 0;
}
