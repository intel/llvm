// RUN: %clangxx -fsycl -fsyntax-only %s

// Tests that identityless reductions compile when applied to a span.

#include <sycl/sycl.hpp>

template <class T> struct PlusWithoutIdentity {
  T operator()(const T &A, const T &B) const { return A + B; }
};

int main() {
  sycl::queue Q;

  int *ScalarMem = sycl::malloc_shared<int>(1, Q);
  int *SpanMem = sycl::malloc_shared<int>(8, Q);
  auto ScalarRed = sycl::reduction(ScalarMem, PlusWithoutIdentity<int>{});
  auto SpanRed = sycl::reduction(sycl::span<int, 8>{SpanMem, 8},
                                 PlusWithoutIdentity<int>{});
  Q.parallel_for(sycl::range<1>{1024}, ScalarRed, SpanRed,
                 [=](sycl::item<1>, auto &, auto &) {});
  Q.parallel_for(sycl::nd_range<1>{1024, 1024}, ScalarRed, SpanRed,
                 [=](sycl::nd_item<1>, auto &, auto &) {});

  return 0;
}
