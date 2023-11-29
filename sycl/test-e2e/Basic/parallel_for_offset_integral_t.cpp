// RUN: %{build} -DLAMBDA_KERNEL=1 -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -o %t1.out
// RUN: %{run} %t1.out
// RUN: %{build} -DLAMBDA_KERNEL=0 -DSYCL2020_DISABLE_DEPRECATION_WARNINGS -o %t2.out
// RUN: %{run} %t2.out

#include <sycl/sycl.hpp>

template <typename AccT> class func {
  AccT acc;

public:
  func(AccT acc) { this->acc = acc; }
  void operator()(size_t ind) const { acc[ind - 1] = ind; }
};

int main() {
  constexpr int arr_size = 16;
  int arr[arr_size];

  {
    sycl::queue q;
    sycl::buffer<int, 1> buf(arr, arr_size);
    q.submit([&](sycl::handler &cgh) {
      auto acc = buf.get_access<sycl::access_mode::write>(cgh);
#if LAMBDA_KERNEL
      cgh.parallel_for(sycl::range<1>(arr_size), sycl::id<1>(1),
                       [=](size_t ind) {
                         func f(acc);
                         f(ind);
                       });
#else
      cgh.parallel_for(sycl::range<1>(arr_size), sycl::id<1>(1), func(acc));
#endif
    });
  }

  for (int i = 0; i < arr_size; i++)
    assert(arr[i] == i + 1);

  return 0;
}
