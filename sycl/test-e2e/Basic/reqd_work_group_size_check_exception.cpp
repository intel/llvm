// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

#define CHECK_INVALID_REQD_WORK_GROUP_SIZE(Dim, ...)                           \
  {                                                                            \
    bool ExceptionThrown = false;                                              \
    std::error_code Errc;                                                      \
    try {                                                                      \
      q.submit([&](sycl::handler &h) {                                         \
        h.parallel_for(sycl::range<Dim>(__VA_ARGS__),                          \
                       [=](sycl::item<Dim> it)                                 \
                           [[sycl::reqd_work_group_size(__VA_ARGS__)]] {});    \
      });                                                                      \
      q.wait();                                                                \
    } catch (sycl::exception & e) {                                            \
      ExceptionThrown = true;                                                  \
      Errc = e.code();                                                         \
    }                                                                          \
    assert(ExceptionThrown &&                                                  \
           "Invalid use of reqd_work_group_size should throw an exception.");  \
    assert(Errc == sycl::errc::kernel_not_supported);                          \
  }

int main() {
  sycl::queue q;
  constexpr int N = 1e9;
  auto MaxWGSize =
      q.get_device().get_info<sycl::info::device::max_work_group_size>();

  if (N > MaxWGSize) {
    CHECK_INVALID_REQD_WORK_GROUP_SIZE(1, N)
    CHECK_INVALID_REQD_WORK_GROUP_SIZE(2, 1, N)
    CHECK_INVALID_REQD_WORK_GROUP_SIZE(3, 1, 1, N)
  }

  return 0;
}
