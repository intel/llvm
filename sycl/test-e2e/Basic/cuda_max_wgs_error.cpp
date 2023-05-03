// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -fno-sycl-id-queries-fit-in-int
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// REQUIRES: cuda

#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

const size_t lsize = 32;
const std::string expected_msg =
    "Number of work-groups exceed limit for dimension ";

template <int N>
void check(range<N> global, range<N> local, bool expect_fail = false) {
  queue q;
  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for(nd_range<N>(global, local), [=](nd_item<N> item) {});
    });
  } catch (nd_range_error e) {
    if (expect_fail) {
      std::string msg = e.what();
      assert(msg.rfind(expected_msg, 0) == 0);
    } else {
      throw e;
    }
  }
}

int main() {
  queue q;
  device d = q.get_device();
  id<1> max_1 = d.get_info<sycl::info::device::ext_oneapi_max_work_groups_1d>();
  check(range<1>(max_1[0] * lsize), range<1>(lsize));
  check(range<1>((max_1[0] + 1) * lsize), range<1>(lsize), true);

  id<2> max_2 = d.get_info<sycl::info::device::ext_oneapi_max_work_groups_2d>();
  check(range<2>(1, max_2[1] * lsize), range<2>(1, lsize));
  check(range<2>(1, (max_2[1] + 1) * lsize), range<2>(1, lsize), true);
  check(range<2>(max_2[0] * lsize, 1), range<2>(lsize, 1));
  check(range<2>((max_2[0] + 1) * lsize, 1), range<2>(lsize, 1), true);

  id<3> max_3 = d.get_info<sycl::info::device::ext_oneapi_max_work_groups_3d>();
  check(range<3>(1, 1, max_3[2] * lsize), range<3>(1, 1, lsize));
  check(range<3>(1, 1, (max_3[2] + 1) * lsize), range<3>(1, 1, lsize), true);
  check(range<3>(1, max_3[1] * lsize, 1), range<3>(1, lsize, 1));
  check(range<3>(1, (max_3[1] + 1) * lsize, 1), range<3>(1, lsize, 1), true);
  check(range<3>(max_3[0] * lsize, 1, 1), range<3>(lsize, 1, 1));
  check(range<3>((max_3[0] + 1) * lsize, 1, 1), range<3>(lsize, 1, 1), true);
}
