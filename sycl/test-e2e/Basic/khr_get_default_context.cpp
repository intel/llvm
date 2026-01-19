// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Test checks that the default context contains all of the root devices that
// are associated with this platform.

#include <algorithm>
#include <sycl/detail/core.hpp>
#include <sycl/platform.hpp>

using namespace sycl;

int main() {
  auto platforms = platform::get_platforms();

  for (const auto &plt : platforms) {
    auto def_ctx_devs = plt.khr_get_default_context().get_devices();
    auto root_devs = plt.get_devices();

    for (const auto &dev : root_devs)
      if (std::find(def_ctx_devs.begin(), def_ctx_devs.end(), dev) ==
          def_ctx_devs.end())
        return 1;
  }

  return 0;
}
