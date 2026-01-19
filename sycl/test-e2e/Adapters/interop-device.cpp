// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// SYCL 2020:
// > The execution environment for a SYCL application has a fixed number of root
// > devices which does not vary as the application executes.
//
// Verify that round-robin conversion of a root device (SYCL->native->SYCL)
// doesn't create a "new" SYCL device that isn't equally comparable to one of
// the root devices in the pre-existin fixed hierarchy.

#include <sycl/backend.hpp>
#include <sycl/detail/core.hpp>

int main() {
  auto root_devices = sycl::device::get_devices();

  for (auto d : root_devices) {
    // TODO: No sycl::device interop support for
    // sycl::backend::ext_oneapi_native_cpu, sycl::backend::ext_oneapi_cuda,
    // sycl::backend::ext_oneapi_hip.
    constexpr sycl::backend backends[] = {sycl::backend::opencl,
                                          sycl::backend::ext_oneapi_level_zero};
    sycl::detail::loop<std::size(backends)>([&](auto be_idx) {
      constexpr auto be = backends[be_idx];
      if (d.get_backend() != be)
        return;

      auto native = sycl::get_native<be>(d);
      auto from_native = sycl::make_device<be>(native);
      assert(d == from_native);
      std::hash<sycl::device> hash;
      assert(hash(d) == hash(from_native));
    });
  }

  return 0;
}
