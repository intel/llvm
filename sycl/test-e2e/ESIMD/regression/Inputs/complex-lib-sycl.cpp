#include <sycl/detail/core.hpp>

sycl::event iota(size_t n, sycl::buffer<int, 1> &buf, sycl::queue &Q) {
  auto HK = [&](sycl::handler &H) {
    sycl::accessor acc_y{buf, H, sycl::write_only};
    auto K = [=](sycl::id<1> id) {
      int *y =
          acc_y.template get_multi_ptr<sycl::access::decorated::yes>().get();
      size_t i = id.get(0);
      y[i] = static_cast<int>(i);
    };
    H.parallel_for(n, K);
  };
  return Q.submit(HK);
}
