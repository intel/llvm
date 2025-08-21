// RUN:  %{build} -o %t.out
// RUN:  %{run} %t.out

// Tests that P2P access is not reported as possible across platforms.

#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>

int main() {
  sycl::device D1;
  sycl::device D2 = D1;
  for (sycl::platform P : sycl::platform::get_platforms()) {
    if (P != D1.get_platform() && !P.get_devices().empty()) {
      D2 = P.get_devices()[0];
      break;
    }
  }

  if (D1 == D2) {
    std::cout << "There are no devices from different platforms. Skipping."
              << std::endl;
    return 0;
  }

  assert(!D1.ext_oneapi_can_access_peer(D2));
  return 0;
}
