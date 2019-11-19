// RUN: %clangxx -fsycl %s -o %t.out
// RUN: env SYCL_DEVICE_TYPE=HOST %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

using namespace cl::sycl;

int main() {
  auto asyncHandler = [](exception_list es) {
    for (auto& e : es) {
      std::rethrow_exception(e);
    }
  };

  // parallel_for, 100 global, 3 local -> fail.
  try {
    queue q(asyncHandler);
    q.submit([&](handler &cgh) {
        cgh.parallel_for<class a>(nd_range<1>(range<1>(100), range<1>(3)),
            [=](nd_item<1> id) {});
    });
    q.wait_and_throw();
    assert(false && "Should have thrown exception");
  } catch (nd_range_error e) {
    // We expect an error to be thrown!
  }

  // parallel_for, 100 global, 4 local -> pass.
  try {
    queue q(asyncHandler);
    q.submit([&](handler &cgh) {
        cgh.parallel_for<class b>(nd_range<1>(range<1>(100), range<1>(4)),
            [=](nd_item<1> id) {});
    });
    q.wait_and_throw();
  } catch (nd_range_error e) {
    assert(false && "Should not have thrown exception");
  }

  // parallel_for, (100, 33, 16) global, (2, 3, 4) local -> pass.
  try {
    queue q(asyncHandler);
    q.submit([&](handler &cgh) {
        cgh.parallel_for<class c>(nd_range<3>(range<3>(100, 33, 16),
              range<3>(2, 3, 4)),
            [=](nd_item<3> id) {});
    });
    q.wait_and_throw();
  } catch (nd_range_error e) {
    assert(false && "Should not have thrown exception");
  }

  // parallel_for, (100, 33, 16) global, (2, 3, 5) local -> fail.
  try {
    queue q(asyncHandler);
    q.submit([&](handler &cgh) {
        cgh.parallel_for<class d>(nd_range<3>(range<3>(100, 33, 16),
              range<3>(2, 3, 5)),
            [=](nd_item<3> id) {});
    });
    q.wait_and_throw();
    assert(false && "Should have thrown exception");
  } catch (nd_range_error e) {
  }

  // local size has a 0-based range -- no SIGFPEs, we hope.
  try {
    queue q(asyncHandler);
    q.submit([&](handler &cgh) {
        cgh.parallel_for<class e>(nd_range<2>(range<2>(5, 33), range<2>(1, 0)),
            [=](nd_item<2> id) {});
    });
    q.wait_and_throw();
    assert(false && "Should have thrown exception");
  } catch (runtime_error e) {
  }

  // parallel_for_work_group with 0-based local range.
  try {
    queue q(asyncHandler);
    q.submit([&](handler &cgh) {
        cgh.parallel_for_work_group<class f>(range<2>(5, 33), range<2>(1, 0),
            [=](group<2> g) {});
    });
    q.wait_and_throw();
    assert(false && "Should have thrown exception");
  } catch (runtime_error e) {
  }
  return 0;
}
