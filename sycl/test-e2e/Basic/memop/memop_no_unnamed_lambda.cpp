// RUN: %{build} -fno-sycl-unnamed-lambda -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#include <sycl/usm.hpp>

// Second test for https://github.com/intel/llvm/issues/469. Verify that the
// mode without unnamed lambdas support still has some limited support.

int main(int argc, char *argv[]) {
  sycl::queue q;
  void *p = sycl::aligned_alloc_device(alignof(int), sizeof(int), q);

  q.fill(p, static_cast<int>(-1), 1).wait();
  // Same sizeof/alignment but different type to ensure no kernel name
  // collisions happen.
  q.fill(p, static_cast<unsigned int>(2), 1).wait();

  sycl::free(p, q);
  return 0;
}
