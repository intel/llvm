// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks sycl::ext::oneapi::property::queue::priority and the priority_range
// device descriptor.

#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/backend/hip.hpp>
#include <sycl/ext/oneapi/info/device.hpp>
#include <sycl/properties/all_properties.hpp>

#include <cassert>
#include <iostream>

namespace prop = sycl::ext::oneapi::property::queue;

static_assert(SYCL_EXT_ONEAPI_QUEUE_PRIORITY >= 2);

static void checkQueue(const sycl::device &Dev,
                       const sycl::property_list &Props) {
  sycl::queue Q{Dev, Props};
  // Some adapters create native queues/streams lazily. Force creation without
  // GPU work that could be starved by other tests:
  // https://github.com/intel/llvm/issues/23735.
  switch (Dev.get_backend()) {
  case sycl::backend::ext_oneapi_level_zero:
    (void)sycl::get_native<sycl::backend::ext_oneapi_level_zero>(Q);
    break;
  case sycl::backend::ext_oneapi_cuda:
    (void)sycl::get_native<sycl::backend::ext_oneapi_cuda>(Q);
    break;
  case sycl::backend::ext_oneapi_hip:
    (void)sycl::get_native<sycl::backend::ext_oneapi_hip>(Q);
    break;
  default:
    break;
  }
}

static void checkThrows(const sycl::device &Dev,
                        const sycl::property_list &Props) {
  try {
    sycl::queue Q{Dev, Props};
  } catch (sycl::exception &E) {
    assert(E.code() == sycl::errc::invalid);
    return;
  }
  assert(false && "Expected an exception");
}

int main() {
  sycl::device Dev;
  auto [Lowest, Highest] =
      Dev.get_info<sycl::ext::oneapi::info::device::priority_range>();
  std::cout << "priority range: lowest=" << Lowest << " highest=" << Highest
            << std::endl;
  assert(Highest <= 0 && Lowest >= 0);

  // Values outside of the range are clamped.
  for (int P = Highest - 1; P <= Lowest + 1; ++P) {
    checkQueue(Dev, {prop::priority{P}});
    checkQueue(Dev, {prop::priority{P}, sycl::property::queue::in_order{}});
  }
  checkQueue(Dev, {prop::priority_normal{}});
  checkQueue(Dev, {prop::priority_low{}});
  checkQueue(Dev, {prop::priority_high{}});

  checkThrows(Dev, {prop::priority{0}, prop::priority_normal{}});
  checkThrows(Dev, {prop::priority{Highest}, prop::priority_high{}});
  checkThrows(Dev, {prop::priority{Lowest}, prop::priority_low{}});

  sycl::queue Q{Dev, {prop::priority{Highest}}};
  assert(Q.has_property<prop::priority>());
  assert(Q.get_property<prop::priority>().get_priority() == Highest);
  return 0;
}
