// UNSUPPORTED: cuda || hip
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -fno-sycl-id-queries-fit-in-int
// RUN: %GPU_RUN_PLACEHOLDER %t.out

#include <numeric>
#include <sycl/sycl.hpp>

using namespace sycl;

const std::string expected_msg =
    "Total number of work-items in a work-group cannot exceed";

template <int N>
void check(range<N> global, range<N> local, bool expect_fail = false) {
  queue q;
  try {
    q.submit([&](handler &cgh) {
      cgh.parallel_for(nd_range<N>(global, local), [=](nd_item<N> item) {});
    });
    assert(!expect_fail && "Exception expected");
  } catch (nd_range_error e) {
    if (expect_fail) {
      std::string msg = e.what();
      assert(msg.rfind(expected_msg, 0) == 0 && "Unexpected error message");
    } else {
      throw e;
    }
  }
}

int main() {
  queue q;
  device d = q.get_device();
  id<2> max_2 = d.get_info<sycl::info::device::max_work_item_sizes<2>>();
  check(range<2>(max_2[0], max_2[1]), range<2>(max_2[0], max_2[1]), true);

  id<3> max_3 = d.get_info<sycl::info::device::max_work_item_sizes<3>>();
  check(range<3>(max_3[0], max_3[1], max_3[2]),
        range<3>(max_3[0], max_3[1], max_3[2]), true);
}
