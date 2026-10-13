// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Checks sycl::ext::oneapi::property::queue::priority and the priority_range
// device descriptor.

#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/info/device.hpp>
#include <sycl/properties/all_properties.hpp>
#include <sycl/usm.hpp>

#include <cassert>
#include <iostream>

namespace prop = sycl::ext::oneapi::property::queue;

static_assert(SYCL_EXT_ONEAPI_QUEUE_PRIORITY >= 2);

static void checkQueue(const sycl::device &Dev,
                       const sycl::property_list &Props) {
  sycl::queue Q{Dev, Props};
  constexpr size_t N = 64;
  int *Data = sycl::malloc_shared<int>(N, Q);
  Q.parallel_for(sycl::range<1>{N}, [=](sycl::id<1> I) {
     Data[I] = static_cast<int>(I[0]);
   }).wait();
  for (size_t I = 0; I < N; ++I)
    assert(Data[I] == static_cast<int>(I));
  sycl::free(Data, Q);
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
