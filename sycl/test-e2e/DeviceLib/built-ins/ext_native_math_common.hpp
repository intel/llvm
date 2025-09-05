#include <cassert>
#include <sycl/detail/core.hpp>
#include <sycl/ext/oneapi/experimental/builtins.hpp>

template <typename T, size_t N>
void assert_out_of_bound(sycl::marray<T, N> val, sycl::marray<T, N> lower,
                         sycl::marray<T, N> upper) {
  for (int i = 0; i < N; i++) {
    assert(lower[i] < val[i] && val[i] < upper[i]);
  }
}

template <typename T> void assert_out_of_bound(T val, T lower, T upper) {
  assert(sycl::all(lower < val && val < upper));
}

template <>
void assert_out_of_bound<float>(float val, float lower, float upper) {
  assert(lower < val && val < upper);
}

template <>
void assert_out_of_bound<sycl::half>(sycl::half val, sycl::half lower,
                                     sycl::half upper) {
  assert(lower < val && val < upper);
}

template <typename T>
void native_tanh_tester(sycl::queue q, T val, T up, T lo) {
  T r = val;

#ifdef SYCL_EXT_ONEAPI_NATIVE_MATH
  {
    sycl::buffer<T, 1> BufR(&r, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      auto AccR = BufR.template get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        AccR[0] = sycl::ext::oneapi::experimental::native::tanh(AccR[0]);
      });
    });
  }

  assert_out_of_bound(r, up, lo);
#else
  assert(!"SYCL_EXT_ONEAPI_NATIVE_MATH not supported");
#endif
}

template <typename T>
void native_exp2_tester(sycl::queue q, T val, T up, T lo) {
  T r = val;

#ifdef SYCL_EXT_ONEAPI_NATIVE_MATH
  {
    sycl::buffer<T, 1> BufR(&r, sycl::range<1>(1));
    q.submit([&](sycl::handler &cgh) {
      auto AccR = BufR.template get_access<sycl::access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        AccR[0] = sycl::ext::oneapi::experimental::native::exp2(AccR[0]);
      });
    });
  }

  assert_out_of_bound(r, up, lo);
#else
  assert(!"SYCL_EXT_ONEAPI_NATIVE_MATH not supported");
#endif
}
