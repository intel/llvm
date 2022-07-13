// UNSUPPORTED: cuda || hip
//
// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out -v
// RUN: %HOST_RUN_PLACEHOLDER %t.out %HOST_CHECK_PLACEHOLDER
// RUN: %CPU_RUN_PLACEHOLDER %t.out %CPU_CHECK_PLACEHOLDER
// RUN: %GPU_RUN_PLACEHOLDER %t.out %GPU_CHECK_PLACEHOLDER
//
// The test checks that multiple usages of the same specialization constant
// works correctly: toolchain processes them correctly and runtime can
// correctly execute the program.
//
// CHECK: --------> 1

#include <sycl/sycl.hpp>

using namespace cl::sycl;

class sc_kernel_t;

namespace test {

struct pod_t {
  float x;
  float y;
};

template <typename T> class kernel_t {
public:
  using sc_t =
      sycl::ext::oneapi::experimental::spec_constant<pod_t, sc_kernel_t>;

  kernel_t(const sc_t &sc, cl::sycl::stream &strm) : sc_(sc), strm_(strm) {}

  void operator()(cl::sycl::id<1> i) const {
    strm_ << "--------> " << sc_.get().x << sycl::endl;
  }

  sc_t sc_;
  cl::sycl::stream strm_;
};

template <typename T> class kernel_driver_t {
public:
  void execute(const pod_t &pod) {
    device dev = sycl::device(default_selector{});
    context ctx = context(dev);
    queue q(dev);

    cl::sycl::program p(q.get_context());
    auto sc = p.set_spec_constant<sc_kernel_t>(pod);
    p.build_with_kernel_type<kernel_t<T>>();

    q.submit([&](cl::sycl::handler &cgh) {
      cl::sycl::stream strm(1024, 256, cgh);
      kernel_t<T> func(sc, strm);

      auto sycl_kernel = p.get_kernel<kernel_t<T>>();
      cgh.parallel_for(sycl_kernel, cl::sycl::range<1>(1), func);
    });
    q.wait();
  }
};

template class kernel_driver_t<float>;

// The line below instantiates the second use of the spec constant named
// `sc_kernel_t`, which used to corrupt the spec constant content
template class kernel_driver_t<int>;
} // namespace test

int main() {
  test::pod_t pod = {1, 2};
  test::kernel_driver_t<float> kd_float;
  kd_float.execute(pod);

  return 0;
}
