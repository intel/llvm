// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

// Check that an exception with an exception with the `errc::invalid` error code
// thrown when trying to use `sycl_ext_oneapi_private_alloca` and no device
// supports the aspect.

#include <sycl/detail/core.hpp>

#include <sycl/ext/oneapi/experimental/alloca.hpp>
#include <sycl/specialization_id.hpp>
#include <sycl/usm.hpp>

class Kernel;

constexpr sycl::specialization_id<int> Size(10);

static std::error_code test() {
  sycl::queue Queue;
  sycl::buffer<int> B(10);

  try {
    Queue.submit([&](sycl::handler &Cgh) {
      sycl::accessor Acc(B, Cgh, sycl::write_only, sycl::no_init);
      Cgh.parallel_for<Kernel>(10, [=](sycl::id<1>, sycl::kernel_handler Kh) {
        sycl::ext::oneapi::experimental::private_alloca<
            int, Size, sycl::access::decorated::no>(Kh);
      });
    });
  } catch (sycl::exception &Exception) {
    return Exception.code();
  }
  assert(false && "Exception not thrown");
}

int main() {
  assert(test() == sycl::errc::invalid && "Unexpected error code");

  return 0;
}
