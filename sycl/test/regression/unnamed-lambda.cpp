// RUN: %clangxx -fsycl -fsyntax-only %s

#include <sycl/sycl.hpp>

// This validates that the unnamed lambda logic in the library correctly works
// with a new implementation of __builtin_unique_stable_name, where
// instantiation order matters.  parallel_for instantiates the KernelInfo before
// the kernel itself, so this checks that example, which only happens when the
// named kernel is inside another lambda.

void foo(sycl::queue queue) {
  sycl::event queue_event2 = queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for<class K1>(sycl::range<1>{1}, [=](sycl::item<1> id) {});
  });
}

// This validates the case when we use a functor instead of lambda, explicitly
// set the kernel name and pass '-fsycl-unnamed-lambda' flag: we had a bug that
// device compiler did not apply unique stable naming scheme for the kernel
// name, because FE didn't mark 'class K2' as kernel name and instead marked
// 'decltype(f)' as kernel name.

class Functor {
public:
  void operator()(sycl::item<2>) const {}
};

void bar(sycl::queue queue) {
  queue.submit([&](sycl::handler &cgh) {
    Functor f;
    cgh.parallel_for<class K2>(sycl::range<2>{1024, 768}, f);
  });
}
