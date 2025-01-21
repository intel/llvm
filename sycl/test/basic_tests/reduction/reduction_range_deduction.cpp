// RUN: %clangxx -fsycl -fsyntax-only %s

// Tests reduction parallel_for can use SYCL 2020 range deduction guides.

#include <sycl/sycl.hpp>

template <class T> struct PlusWithoutIdentity {
  T operator()(const T &A, const T &B) const { return A + B; }
};

int main() {
  sycl::queue Q;

  int *ScalarMem = sycl::malloc_shared<int>(1, Q);
  int *SpanMem = sycl::malloc_shared<int>(8, Q);
  auto ScalarRed1 = sycl::reduction(ScalarMem, std::plus<int>{});
  auto ScalarRed2 = sycl::reduction(ScalarMem, PlusWithoutIdentity<int>{});
  auto SpanRed1 =
      sycl::reduction(sycl::span<int, 8>{SpanMem, 8}, std::plus<int>{});
  auto SpanRed2 = sycl::reduction(sycl::span<int, 8>{SpanMem, 8},
                                  PlusWithoutIdentity<int>{});

  // Shortcut and range<1> deduction from integer.
  Q.parallel_for(1024, ScalarRed1, [=](sycl::item<1>, auto &) {});
  Q.parallel_for(1024, SpanRed1, [=](sycl::item<1>, auto &) {});
  Q.parallel_for(1024, ScalarRed1, ScalarRed2,
                 [=](sycl::item<1>, auto &, auto &) {});
  Q.parallel_for(1024, SpanRed1, SpanRed2,
                 [=](sycl::item<1>, auto &, auto &) {});
  Q.parallel_for(1024, ScalarRed1, SpanRed1, ScalarRed2, SpanRed2,
                 [=](sycl::item<1>, auto &, auto &, auto &, auto &) {});

  // Shortcut and range<1> deduction from initializer.
  Q.parallel_for({1024}, ScalarRed1, [=](sycl::item<1>, auto &) {});
  Q.parallel_for({1024}, SpanRed1, [=](sycl::item<1>, auto &) {});
  Q.parallel_for({1024}, ScalarRed1, ScalarRed2,
                 [=](sycl::item<1>, auto &, auto &) {});
  Q.parallel_for({1024}, SpanRed1, SpanRed2,
                 [=](sycl::item<1>, auto &, auto &) {});
  Q.parallel_for({1024}, ScalarRed1, SpanRed1, ScalarRed2, SpanRed2,
                 [=](sycl::item<1>, auto &, auto &, auto &, auto &) {});

  // Shortcut and range<2> deduction from initializer.
  Q.parallel_for({1024, 1024}, ScalarRed1, [=](sycl::item<2>, auto &) {});
  Q.parallel_for({1024, 1024}, SpanRed1, [=](sycl::item<2>, auto &) {});
  Q.parallel_for({1024, 1024}, ScalarRed1, ScalarRed2,
                 [=](sycl::item<2>, auto &, auto &) {});
  Q.parallel_for({1024, 1024}, SpanRed1, SpanRed2,
                 [=](sycl::item<2>, auto &, auto &) {});
  Q.parallel_for({1024, 1024}, ScalarRed1, SpanRed1, ScalarRed2, SpanRed2,
                 [=](sycl::item<2>, auto &, auto &, auto &, auto &) {});

  // Shortcut and range<3> deduction from initializer.
  Q.parallel_for({1024, 1024, 1024}, ScalarRed1, [=](sycl::item<3>, auto &) {});
  Q.parallel_for({1024, 1024, 1024}, SpanRed1, [=](sycl::item<3>, auto &) {});
  Q.parallel_for({1024, 1024, 1024}, ScalarRed1, ScalarRed2,
                 [=](sycl::item<3>, auto &, auto &) {});
  Q.parallel_for({1024, 1024, 1024}, SpanRed1, SpanRed2,
                 [=](sycl::item<3>, auto &, auto &) {});
  Q.parallel_for({1024, 1024, 1024}, ScalarRed1, SpanRed1, ScalarRed2, SpanRed2,
                 [=](sycl::item<3>, auto &, auto &, auto &, auto &) {});

  // Submission and range<1> deduction from integer.
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(1024, ScalarRed1, [=](sycl::item<1>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(1024, SpanRed1, [=](sycl::item<1>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(1024, ScalarRed1, ScalarRed2,
                     [=](sycl::item<1>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(1024, SpanRed1, SpanRed2,
                     [=](sycl::item<1>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for(1024, ScalarRed1, SpanRed1, ScalarRed2, SpanRed2,
                     [=](sycl::item<1>, auto &, auto &, auto &, auto &) {});
  });

  // Submission and range<1> deduction from initializer.
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024}, ScalarRed1, [=](sycl::item<1>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024}, SpanRed1, [=](sycl::item<1>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024}, ScalarRed1, ScalarRed2,
                     [=](sycl::item<1>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024}, SpanRed1, SpanRed2,
                     [=](sycl::item<1>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024}, ScalarRed1, SpanRed1, ScalarRed2, SpanRed2,
                     [=](sycl::item<1>, auto &, auto &, auto &, auto &) {});
  });

  // Submission and range<2> deduction from initializer.
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024}, ScalarRed1, [=](sycl::item<2>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024}, SpanRed1, [=](sycl::item<2>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024}, ScalarRed1, ScalarRed2,
                     [=](sycl::item<2>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024}, SpanRed1, SpanRed2,
                     [=](sycl::item<2>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024}, ScalarRed1, SpanRed1, ScalarRed2, SpanRed2,
                     [=](sycl::item<2>, auto &, auto &, auto &, auto &) {});
  });

  // Submission and range<3> deduction from initializer.
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024, 1024}, ScalarRed1,
                     [=](sycl::item<3>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024, 1024}, SpanRed1,
                     [=](sycl::item<3>, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024, 1024}, ScalarRed1, ScalarRed2,
                     [=](sycl::item<3>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024, 1024}, SpanRed1, SpanRed2,
                     [=](sycl::item<3>, auto &, auto &) {});
  });
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for({1024, 1024, 1024}, ScalarRed1, SpanRed1, ScalarRed2,
                     SpanRed2,
                     [=](sycl::item<3>, auto &, auto &, auto &, auto &) {});
  });

  return 0;
}
