// UNSUPPORTED: hip
// RUN: %{build} -o %t.out -fno-sycl-id-queries-fit-in-int
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;
queue q;
int n_fail = 0;

template <typename FunctorT>
void throws_kernel_not_supported(const char *test_name, FunctorT f) {
  try {
    f();
  } catch (const sycl::exception &e) {
    if (e.code() != errc::kernel_not_supported) {
      std::cout << "fail: " << test_name << "\n"
                << "Caught wrong exception with error code " << e.code() << "\n"
                << e.what() << "\n";
      ++n_fail;
      return;
    } else {
      std::cout << "pass: " << test_name << "\n"
                << "Caught right exception:\n"
                << e.what() << "\n";
      return;
    }
  }
  std::cout << "fail: " << test_name << "\n"
            << "No exception thrown\n";
  ++n_fail;
  return;
}

int main(int argc, char *argv[]) {
  throws_kernel_not_supported("nd_range<1>", [] {
    constexpr uint32_t N = std::numeric_limits<uint32_t>::max();
    q.parallel_for<class K0>(nd_range<1>(N, N),
                             [=](auto) [[sycl::reqd_work_group_size(N)]] {});
  });

  throws_kernel_not_supported("nd_range<2>", [] {
    constexpr uint32_t N = std::numeric_limits<uint32_t>::max();
    q.parallel_for<class K1>(nd_range<2>({N, N}, {N, N}),
                             [=](auto) [[sycl::reqd_work_group_size(N, N)]] {});
  });

  throws_kernel_not_supported("nd_range<3>", [] {
    constexpr uint32_t N = std::numeric_limits<uint32_t>::max();
    q.parallel_for<class K2>(nd_range<3>({N, N, N}, {N, N, N}),
                             [=](auto)
                                 [[sycl::reqd_work_group_size(N, N, N)]] {});
  });

  throws_kernel_not_supported("uint32_max+2", [] {
    constexpr uint64_t N = std::numeric_limits<uint32_t>::max() + uint64_t(2);
    q.parallel_for<class K3>(nd_range<1>(N, N),
                             [=](auto) [[sycl::reqd_work_group_size(N)]] {});
  });

  return n_fail;
}
