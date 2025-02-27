// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Ensure that `core.hpp` has necessary includes for the "recommended" (in-order
// quee/USM/event-less submit) way to program in SYCL to get the least overhead
// possible.

#include <sycl/detail/core.hpp>

int main() {
  sycl::queue q{sycl::property::queue::in_order{}};

  using sycl::ext::oneapi::experimental::usm_deleter;

  std::unique_ptr<int, usm_deleter> res{sycl::malloc_shared<int>(1, q), {q}};
  sycl::ext::oneapi::experimental::single_task(
      q, [res = res.get()]() { *res = 42; });
  q.wait();
  assert(*res == 42);
}
