// REQUIRES: level_zero, aspect-usm_device_allocations
// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env ZE_FLAT_DEVICE_HIERARCHY=COMPOSITE %{run} %t.out
//
// A bundle built for a sub-device inside a root-device context. Level Zero
// lists only the root device on the program, so the built binary cannot be
// matched to the sub-device; the query must still hand back usable content
// rather than assert. Skips when the device cannot be partitioned.

#include <sycl/detail/core.hpp>
#include <sycl/kernel_bundle.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <iostream>
#include <vector>

class SubDevKernel;

int main() {
  sycl::device Root;
  std::vector<sycl::device> Subs;
  try {
    Subs = Root.create_sub_devices<
        sycl::info::partition_property::partition_by_affinity_domain>(
        sycl::info::partition_affinity_domain::next_partitionable);
  } catch (const sycl::exception &) {
    std::cout << "Device has no sub-devices, skipping" << std::endl;
    return 0;
  }
  if (Subs.empty())
    return 0;

  sycl::context Ctx{Root};
  sycl::queue Q{Ctx, Subs[0]};

  int *Ptr = sycl::malloc_device<int>(1, Q);
  Q.single_task<SubDevKernel>([=] { *Ptr = 42; }).wait();
  int Host = 0;
  Q.memcpy(&Host, Ptr, sizeof(int)).wait();
  assert(Host == 42);
  sycl::free(Ptr, Q);

  auto KB = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      Ctx, {Subs[0]}, {sycl::get_kernel_id<SubDevKernel>()});
  assert(!KB.empty());
  for (const auto &Img : KB) {
    std::vector<std::byte> Bytes = Img.ext_oneapi_get_backend_content();
    assert(!Bytes.empty());
#ifdef __cpp_lib_span
    auto View = Img.ext_oneapi_get_backend_content_view();
    assert(View.size() == Bytes.size());
#endif
  }
  std::cout << "Done" << std::endl;
  return 0;
}
