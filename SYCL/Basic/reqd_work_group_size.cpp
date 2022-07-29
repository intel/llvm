// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out
//
// Failing negative test with HIP
// XFAIL: hip

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);
  device D(Q.get_device());

  bool IsOpenCL = (D.get_platform().get_backend() == backend::opencl);

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(4, 4, 4) ->
  // pass
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class ReqdWGSizePositiveA>(
          nd_range<3>(range<3>(8, 8, 8), range<3>(4, 4, 4)), [=
      ](nd_item<3>) [[sycl::reqd_work_group_size(4, 4, 4)]]{});
    });
    Q.wait_and_throw();
  } catch (nd_range_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected "
                 "nd_range_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected "
                 "runtime_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  // Kernel that has a required WG size, but no local size is specified.
  //
  // TODO: This fails on OpenCL and should be investigated.
  if (!IsOpenCL) {
    try {
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<class ReqdWGSizeNoLocalPositive>(
            range<3>(16, 16, 16), [=
        ](item<3>) [[sycl::reqd_work_group_size(4, 4, 4)]]{});
      });
      Q.wait_and_throw();
    } catch (nd_range_error &E) {
      std::cerr << "Test case ReqdWGSizeNoLocalPositive failed: unexpected "
                   "nd_range_error exception: "
                << E.what() << std::endl;
      return 1;
    } catch (runtime_error &E) {
      std::cerr
          << "Test case ReqdWGSizeNoLocalPositive: unexpected runtime_error "
             "exception: "
          << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case ReqdWGSizeNoLocalPositive failed: something "
                   "unexpected has been caught"
                << std::endl;
      return 1;
    }
  }

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (16, 16, 16) global, (8, 8, 8) local, reqd_wg_size(4, 4, 4)
  // -> fail
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class ReqdWGSizeNegativeA>(
          nd_range<3>(range<3>(16, 16, 16), range<3>(8, 8, 8)), [=
      ](nd_item<3>) [[sycl::reqd_work_group_size(4, 4, 4)]]{

                                                                });
    });
    Q.wait_and_throw();
    std::cerr << "Test case ReqdWGSizeNegativeA failed: no exception has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (nd_range_error &E) {
    if (std::string(E.what()).find(
            "The specified local size {8, 8, 8} doesn't match the required "
            "work-group size specified in the program source {4, 4, 4}") ==
        std::string::npos) {
      std::cerr
          << "Test case ReqdWGSizeNegativeA failed: unexpected nd_range_error "
             "exception: "
          << E.what() << std::endl;
      return 1;
    }
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed: unexpected "
                 "nd_range_error exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  return 0;
}
