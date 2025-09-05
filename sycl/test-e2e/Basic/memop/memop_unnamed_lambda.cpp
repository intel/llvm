// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

// Test for https://github.com/intel/llvm/issues/469.

#if !defined(__SYCL_UNNAMED_LAMBDA__)
#error "This test verifies unnamed lambda code path!"
#endif

int main(int argc, char *argv[]) {
  struct Simple {
    int a;
  };

  // Same layout but different name to ensure no kernel name collisions happen.
  struct Simple2 {
    int c;
  };

  sycl::queue q;
  void *p = sycl::aligned_alloc_device(alignof(Simple), sizeof(Simple), q);

  q.fill(p, Simple{1}, 1).wait();
  q.fill(p, Simple2{2}, 1).wait();

  sycl::free(p, q);
  return 0;
}
