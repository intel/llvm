// RUN: %{build} -fsyntax-only -o %t.out

#include <sycl/detail/kernel_desc.hpp>
#include <sycl/queue.hpp>

using namespace sycl;

class A;
class B;

int main() {

  queue Queue;

  // No special captures; only values and pointers.
  int Value;
  int *Pointer;
  Queue.parallel_for<A>(nd_range<1>{1, 1},
                        [=](nd_item<1> Item) { *Pointer += Value; });
#ifndef __SYCL_DEVICE_ONLY__
  static_assert(!detail::hasSpecialCaptures<A>());
#endif

  // An accessor is a special capture.
  accessor<int> Accessor;
  Queue.parallel_for<B>(nd_range<1>{1, 1}, [=](nd_item<1> Item) {
    *Pointer += Value;
    Accessor[0] += Value;
  });
#ifndef __SYCL_DEVICE_ONLY__
  static_assert(detail::hasSpecialCaptures<B>());
#endif
}
