// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsyntax-only -Wno-sycl-2017-compat -verify %s
// RUN: %clang_cc1 -fsycl -fsycl-is-device -fsycl-unnamed-lambda -fsyntax-only -Wno-sycl-2017-compat -verify %s
#include "Inputs/sycl.hpp"

#ifdef __SYCL_UNNAMED_LAMBDA__
// expected-no-diagnostics
#endif

namespace namespace1 {
template <typename T>
class KernelName;
}

namespace std {
typedef struct {
} max_align_t;
} // namespace std

template <typename T>
struct Templated_kernel_name;

template <typename T>
struct Templated_kernel_name2;

struct MyWrapper {
private:
  class InvalidKernelName0 {};
  class InvalidKernelName3 {};
  class InvalidKernelName4 {};
  class InvalidKernelName5 {};

public:
  void test() {
    cl::sycl::queue q;

    class InvalidKernelName1 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName1>([] {});
    });

    class InvalidKernelName2 {};
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName2>>([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidKernelName0>([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidKernelName3>>([] {});
    });

    using ValidAlias = MyWrapper;
    q.submit([&](cl::sycl::handler &h) {
      h.single_task<ValidAlias>([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<std::max_align_t>([] {});
    });

    using InvalidAlias = InvalidKernelName4;

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<InvalidAlias>([] {});
    });

    using InvalidAlias1 = InvalidKernelName5;

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<namespace1::KernelName<InvalidAlias1>>([] {});
    });

    q.submit([&](cl::sycl::handler &h) {
      h.single_task<Templated_kernel_name2<Templated_kernel_name<InvalidKernelName1>>>([] {});
    });
  }
};

int main() {
  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &h) { h.single_task([] {}); });

  return 0;
}
