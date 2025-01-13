// RUN: %{build} -o %t.out
// RUN: env ONEAPI_DEVICE_SELECTOR=bad %{run-unfiltered-devices} %t.out

#include <sycl/detail/core.hpp>

int main(void) {
  try {
    // This test intentionally uses a bad ONEAPI_DEVICE_SELECTOR
    // By virtue of that, this default constructed event should throw.
    sycl::event e{};

    assert(false && "we should not be here");

    auto be = e.get_backend();
  } catch (std::exception const &e) {
    std::cout << "exception successfully thrown." << e.what() << std::endl;
  } catch (...) {
    assert(false && "we should not be here");
  }
  return 0;
}
