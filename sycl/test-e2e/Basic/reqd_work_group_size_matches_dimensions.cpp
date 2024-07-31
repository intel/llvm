// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

using namespace sycl;

#define CHECK_INVALID_REQD_WORK_GROUP_SIZE(Range, Item, ...)                   \
  {                                                                            \
    bool ExceptionThrown = false;                                              \
    std::error_code Errc;                                                      \
    try {                                                                      \
      q.submit([&](sycl::handler &h) {                                         \
        h.parallel_for(                                                        \
            Range, [=](Item) [[sycl::reqd_work_group_size(__VA_ARGS__)]] {});  \
      });                                                                      \
      q.wait();                                                                \
    } catch (sycl::exception & e) {                                            \
      ExceptionThrown = true;                                                  \
      Errc = e.code();                                                         \
    }                                                                          \
    assert(ExceptionThrown &&                                                  \
           "Invalid use of reqd_work_group_size should throw an exception.");  \
    assert(Errc == sycl::errc::nd_range);                                      \
  }

int main() {
  queue q;
  range<1> range1D(1);
  range<2> range2D(1, 1);
  range<3> range3D(1, 1, 1);

  CHECK_INVALID_REQD_WORK_GROUP_SIZE(range1D, item<1> it, 1, 1)
  CHECK_INVALID_REQD_WORK_GROUP_SIZE(range1D, item<1> it, 1, 1, 1)

  CHECK_INVALID_REQD_WORK_GROUP_SIZE(range2D, item<2> it, 1)
  CHECK_INVALID_REQD_WORK_GROUP_SIZE(range2D, item<2> it, 1, 1, 1)

  CHECK_INVALID_REQD_WORK_GROUP_SIZE(range3D, item<3> it, 1)
  CHECK_INVALID_REQD_WORK_GROUP_SIZE(range3D, item<3> it, 1, 1)

  CHECK_INVALID_REQD_WORK_GROUP_SIZE(nd_range(range1D, range1D), nd_item<1> it,
                                     1, 1)
  CHECK_INVALID_REQD_WORK_GROUP_SIZE(nd_range(range1D, range1D), nd_item<1> it,
                                     1, 1, 1)

  CHECK_INVALID_REQD_WORK_GROUP_SIZE(nd_range(range2D, range2D), nd_item<2> it,
                                     1)
  CHECK_INVALID_REQD_WORK_GROUP_SIZE(nd_range(range2D, range2D), nd_item<2> it,
                                     1, 1, 1)

  CHECK_INVALID_REQD_WORK_GROUP_SIZE(nd_range(range3D, range3D), nd_item<3> it,
                                     1)
  CHECK_INVALID_REQD_WORK_GROUP_SIZE(nd_range(range3D, range3D), nd_item<3> it,
                                     1, 1)
}
