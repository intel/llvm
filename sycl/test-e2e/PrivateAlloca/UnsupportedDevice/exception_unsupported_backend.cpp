// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check that an exception with an exception with the `errc::invalid` error code
// thrown when trying to use `sycl_ext_oneapi_private_alloca` and no device
// supports the aspect.

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

template <int I> class Kernel;

constexpr sycl::specialization_id<int> Size(10);

template <int I, typename Func> static void test(Func f) {
  constexpr size_t N = 10;
  sycl::queue Queue;
  sycl::buffer<int> B(N);

  try {
    Queue.submit([&](sycl::handler &Cgh) {
      sycl::accessor Acc(B, Cgh, sycl::write_only, sycl::no_init);
      Cgh.parallel_for<Kernel<I>>(
          N, [=](sycl::id<1>, sycl::kernel_handler Kh) { f(Kh); });
    });
  } catch (sycl::exception &Exception) {
    assert(Exception.code() == sycl::errc::invalid && "Unexpected error code");
    return;
  }
  assert(false && "Exception not thrown");
}

int main() {
  test<0>([](sycl::kernel_handler &Kh) {
    sycl::ext::oneapi::experimental::private_alloca<
        int, Size, sycl::access::decorated::no>(Kh);
  });

  test<1>([](sycl::kernel_handler &Kh) {
    sycl::ext::oneapi::experimental::aligned_private_alloca<
        int, alignof(int64_t), Size, sycl::access::decorated::no>(Kh);
  });

  return 0;
}
