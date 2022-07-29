// UNSUPPORTED: cuda || hip
//
// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
//
// This test is intended to check that unpacked composites with elemements of
// various sizes are handled correctly
//
// CHECK: --------> 1
// CHECK: --------> 2
// CHECK: --------> 3
// CHECK: --------> 4
#include <sycl/sycl.hpp>

#include <stdint.h>

using namespace sycl;

class sc_kernel_t;

namespace test {

struct pod_t {
  int a;
  int8_t b;
  int c;
  int64_t d;
};

template <typename T> class kernel_t {
public:
  using sc_t =
      sycl::ext::oneapi::experimental::spec_constant<pod_t, sc_kernel_t>;

  kernel_t(const sc_t &sc, sycl::stream &strm) : sc_(sc), strm_(strm) {}

  void operator()(sycl::id<1> i) const {
    strm_ << "--------> " << sc_.get().a << sycl::endl;
    strm_ << "--------> " << sc_.get().b << sycl::endl;
    strm_ << "--------> " << sc_.get().c << sycl::endl;
    strm_ << "--------> " << sc_.get().d << sycl::endl;
  }

  sc_t sc_;
  sycl::stream strm_;
};

template <typename T> class kernel_driver_t {
public:
  void execute(const pod_t &pod) {
    device dev = sycl::device(default_selector{});
    context ctx = context(dev);
    queue q(dev);

    sycl::program p(q.get_context());
    auto sc = p.set_spec_constant<sc_kernel_t>(pod);
    p.build_with_kernel_type<kernel_t<T>>();

    q.submit([&](sycl::handler &cgh) {
      sycl::stream strm(1024, 256, cgh);
      kernel_t<T> func(sc, strm);

      auto sycl_kernel = p.get_kernel<kernel_t<T>>();
      cgh.parallel_for(sycl_kernel, sycl::range<1>(1), func);
    });
    q.wait();
  }
};

} // namespace test

int main() {
  test::pod_t pod = {1, 2, 3, 4};
  test::kernel_driver_t<float> kd_float;
  kd_float.execute(pod);

  return 0;
}
