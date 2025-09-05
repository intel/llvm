// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// SYCL dependencies (i.e. low level runtimes) may have global objects of their
// own. The test ensures, that those objects do not cause problems. When host
// queue is created, no runtimes are loaded. When another device is created,
// low level runtime is dlopen'ed and global objects are created and destructors
// are registered. This can be a potential problem, as many implementations
// use reverse order to call destructors, and low level runtime's objects are
// destroyed before global queue in user code.

#include <sycl/detail/core.hpp>

sycl::queue Queue;

int main() {
  Queue = sycl::queue{sycl::default_selector_v};

  return 0;
}
