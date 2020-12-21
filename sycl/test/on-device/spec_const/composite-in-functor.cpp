// UNSUPPORTED: cuda
//
// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %RUN_ON_HOST %t.out | FileCheck %s
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

#include <CL/sycl.hpp>

using namespace cl::sycl;

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
      sycl::ONEAPI::experimental::spec_constant<pod_t, class my_kernel_t>;

  my_kernel_t(const sc_t &sc, const cl::sycl::stream &strm)
      : sc_(sc), strm_(strm) {}

  void operator()(cl::sycl::id<1> i) const {
    auto p = sc_.get();
    strm_ << p.f1[0] << " : " << p.f1[1] << "\n";
    strm_ << p.f2.x << "\n";
    strm_ << p.f2.y[0] << " : " << p.f2.y[1] << "\n";
    strm_ << sycl::endl;
  }

  sc_t sc_;
  cl::sycl::stream strm_;
};

int main() {
  cl::sycl::queue q(default_selector{}, [](exception_list l) {
    for (auto ep : l) {
      try {
        std::rethrow_exception(ep);
      } catch (cl::sycl::exception &e0) {
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

  cl::sycl::program p(q.get_context());
  auto sc = p.set_spec_constant<my_kernel_t>(pod);
  p.build_with_kernel_type<my_kernel_t>();

  q.submit([&](cl::sycl::handler &cgh) {
    cl::sycl::stream strm(1024, 256, cgh);
    my_kernel_t func(sc, strm);

    auto sycl_kernel = p.get_kernel<my_kernel_t>();
    cgh.parallel_for(sycl_kernel, cl::sycl::range<1>(1), func);
  });
  q.wait();

  return 0;
}
