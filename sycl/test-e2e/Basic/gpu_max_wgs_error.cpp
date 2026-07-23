// REQUIRES: gpu

// The runtime does throw errc::nd_range for an oversized work-group on CUDA,
// but for the work-group size used here (max_work_item_sizes in every
// dimension) the CUDA driver rejects the launch as
// UR_RESULT_ERROR_OUT_OF_RESOURCES (register exhaustion) rather than
// UR_RESULT_ERROR_INVALID_WORK_GROUP_SIZE. That is handled by
// handleOutOfResources(), which throws an nd_range exception with a different,
// register-count message. The "Total number of work-items in a work-group
// cannot exceed" branch added in handleInvalidWorkGroupSize() is only reached
// when each dimension is within limits but their product exceeds the maximum
// work-group size, so this generic test cannot assert that message on CUDA.
// UNSUPPORTED: cuda
// UNSUPPORTED-TRACKER: https://github.com/intel/llvm/issues/22300

// RUN: %{build} -o %t.out -fno-sycl-id-queries-fit-in-int
// RUN: %{run} %t.out

#include <numeric>
#include <sycl/detail/core.hpp>

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
  } catch (exception e) {
    if (expect_fail) {
      std::string msg = e.what();
      assert(e.code() == errc::nd_range);
      assert(msg.rfind(expected_msg, 0) == 0 && "Unexpected error message");
    } else {
      throw e;
    }
  }
}

int main() {
  queue q;
  device d = q.get_device();
  range<2> max_2 = d.get_info<sycl::info::device::max_work_item_sizes<2>>();
  check(max_2, max_2, true);

  range<3> max_3 = d.get_info<sycl::info::device::max_work_item_sizes<3>>();
  check(max_3, max_3, true);
}
