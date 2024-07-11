// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=level_zero:0 ZE_FLAT_DEVICE_HIERARCHY=COMBINED %t.out
// REQUIRES: level_zero

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/composite_device.hpp>

#ifdef SYCL_EXT_ONEAPI_COMPOSITE_DEVICE

using namespace sycl::ext::oneapi::experimental;

int main() {
  sycl::queue q;
  auto Platforms = sycl::platform::get_platforms();

  // Check that we do not expose a composite device unless it represents all of
  // the tiles on a card. Since we are setting ONEAPI_DEVICE_SELECTOR to use
  // only a single tile, both get_composite_devices() and
  // platform::ext_oneapi_get_composite_devices() should return an empty vector.
  std::vector<sycl::device> AllCompositeDevs = get_composite_devices();
  assert(AllCompositeDevs.empty());
  for (const auto &P : Platforms) {
    auto CompositeDevs = P.ext_oneapi_get_composite_devices();
    assert(CompositeDevs.empty());
  }
}

#endif // SYCL_EXT_ONEAPI_COMPOSITE_DEVICE
