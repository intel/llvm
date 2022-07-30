// UNSUPPORTED: cuda || hip
//
// RUN: %clangxx -fsycl %s -D__SYCL_INTERNAL_API -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
//
// The test checks that the specialization constant feature works correctly with
// composite types: toolchain processes them correctly and runtime can correctly
// execute the program.
//
// CHECK: 1 : 2
// CHECK-NEXT: 3
// CHECK-NEXT: 4 : 5

#include <iostream>
#include <sycl/sycl.hpp>

using namespace sycl;

struct A {
  float x;
  float y[2];
};

struct pod_t {
  int f1[2];
  A f2;
};

class my_kernel_t {
public:
  using sc_t =
      sycl::ext::oneapi::experimental::spec_constant<pod_t, class my_kernel_t>;

  my_kernel_t(const sc_t &sc, const sycl::stream &strm)
      : sc_(sc), strm_(strm) {}

  void operator()(sycl::id<1> i) const {
    auto p = sc_.get();
    strm_ << p.f1[0] << " : " << p.f1[1] << "\n";
    strm_ << p.f2.x << "\n";
    strm_ << p.f2.y[0] << " : " << p.f2.y[1] << "\n";
    strm_ << sycl::endl;
  }

  sc_t sc_;
  sycl::stream strm_;
};

int main() {
  sycl::queue q(default_selector{}, [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (sycl::exception &e0) {
        std::cout << e0.what();
      } catch (std::exception &e1) {
        std::cout << e1.what();
      } catch (...) {
        std::cout << "*** catch (...)\n";
      }
    }
  });

  pod_t pod;
  pod.f1[0] = 1;
  pod.f1[1] = 2;
  pod.f2.x = 3;
  pod.f2.y[0] = 4;
  pod.f2.y[1] = 5;

  sycl::program p(q.get_context());
  auto sc = p.set_spec_constant<my_kernel_t>(pod);
  p.build_with_kernel_type<my_kernel_t>();

  q.submit([&](sycl::handler &cgh) {
    sycl::stream strm(1024, 256, cgh);
    my_kernel_t func(sc, strm);

    auto sycl_kernel = p.get_kernel<my_kernel_t>();
    cgh.parallel_for(sycl_kernel, sycl::range<1>(1), func);
  });
  q.wait();

  return 0;
}
