// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

// SYCL dependencies (i.e. low level runtimes) may have global objects of their
// own. The test ensures, that those objects do not cause problems. When host
// queue is created, no runtimes are loaded. When another device is created,
// low level runtime is dlopen'ed and global objects are created and destructors
// are registered. This can be a potential problem, as many implementations
// use reverse order to call destructors, and low level runtime's objects are
// destroyed before global queue in user code.

#include <sycl/sycl.hpp>

sycl::queue Queue;

int main() {
  Queue = sycl::queue{sycl::default_selector{}.select_device()};

  return 0;
}
