// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
// RUN: env SYCL_ENABLE_DEFAULT_CONTEXTS=1 %t.out
// RUN: env SYCL_ENABLE_DEFAULT_CONTEXTS=0 %t.out 1

#include <sycl/detail/core.hpp>

// when not using the environment variable, we use the "default context" on both
// Lin and Win.  This test asserts it defaults correctly, and that the
// environment variable is working as expected.

// no args: YES default context.
// any arg: NO default context.
int main(int argc, char *argv[]) {
  sycl::queue q1;
  sycl::queue q2;

  if (argc <= 1)
    assert(q1.get_context() == q2.get_context());
  else
    assert(q1.get_context() != q2.get_context());
}