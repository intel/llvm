// XFAIL: cuda || opencl
// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);
  device D(Q.get_device());

  string_class DeviceVendorName = D.get_info<info::device::vendor>();
  auto DeviceType = D.get_info<info::device::device_type>();

  // parallel_for, (16, 16, 16) global, (8, 8, 8) local, reqd_wg_size(4, 4, 4)
  // -> fail
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class ReqdWGSizeNegativeA>(
          nd_range<3>(range<3>(16, 16, 16), range<3>(8, 8, 8)),
          [=](nd_item<3>) [[intel::reqd_work_group_size(4, 4, 4)]]{

          });
    });
    Q.wait_and_throw();
    std::cerr << "Test case ReqdWGSizeNegativeA failed: no exception has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (nd_range_error &E) {
    if (string_class(E.what()).find(
            "Specified local size doesn't match the required work-group size "
            "specified in the program source") == string_class::npos) {
      std::cerr
          << "Test case ReqdWGSizeNegativeA failed 1: unexpected exception: "
          << E.what() << std::endl;
      return 1;
    }
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed 2: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  // Positive test-cases that should pass on any underlying OpenCL runtime
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(4, 4, 4) ->
  // pass
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class ReqdWGSizePositiveA>(
          nd_range<3>(range<3>(8, 8, 8), range<3>(4, 4, 4)),
          [=](nd_item<3>) [[intel::reqd_work_group_size(4, 4, 4)]]{});
    });
    Q.wait_and_throw();
  } catch (nd_range_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class ReqdWGSizePositiveB>(
          range<3>(16, 16, 16), [=](item<3>) [[intel::reqd_work_group_size(4, 4, 4)]]{});
    });
    Q.wait_and_throw();

  } catch (nd_range_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveB failed 1: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  } catch (runtime_error &E) {
    std::cerr
        << "Test case ReqdWGSizePositiveB failed 2: unexpected exception: "
        << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizePositiveB failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  return 0;
}
