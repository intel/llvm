// RUN: %{build} -o %t.out
// RUN: %{run} %t.out
//
// Failing negative test with HIP
// UNSUPPORTED: hip

#include <sycl/detail/core.hpp>

#include <iostream>

using namespace sycl;

template <int Dims> std::string GetRangeStr(range<Dims> Range);
template <> std::string GetRangeStr<1>(range<1> Range) {
  return "{" + std::to_string(Range[0]) + "}";
}
template <> std::string GetRangeStr<2>(range<2> Range) {
  return "{" + std::to_string(Range[0]) + "," + std::to_string(Range[1]) + "}";
}
template <> std::string GetRangeStr<3>(range<3> Range) {
  return "{" + std::to_string(Range[0]) + "," + std::to_string(Range[1]) + "," +
         std::to_string(Range[2]) + "}";
}

enum class TestKind : char {
  PositiveWithLocal,
  PositiveWithoutLocal,
  Negative
};

template <TestKind Kind, int... ReqdWGSizes> class ReqdWGSizeVariadic;

template <TestKind Kind, int ReqdWGSizeX, int ReqdWGSizeY, int ReqdWGSizeZ>
void submitTest(queue &Q, range<3> GlobalRange, range<3> LocalRange) {
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<
        ReqdWGSizeVariadic<Kind, ReqdWGSizeX, ReqdWGSizeY, ReqdWGSizeZ>>(
        nd_range<3>(GlobalRange, LocalRange),
        [=](nd_item<3>) [[sycl::reqd_work_group_size(ReqdWGSizeX, ReqdWGSizeY,
                                                     ReqdWGSizeZ)]] {});
  });
}

template <TestKind Kind, int ReqdWGSizeX, int ReqdWGSizeY>
void submitTest(queue &Q, range<2> GlobalRange, range<2> LocalRange) {
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<ReqdWGSizeVariadic<Kind, ReqdWGSizeX, ReqdWGSizeY>>(
        nd_range<2>(GlobalRange, LocalRange),
        [=](nd_item<2>)
            [[sycl::reqd_work_group_size(ReqdWGSizeX, ReqdWGSizeY)]] {});
  });
}

template <TestKind Kind, int ReqdWGSizeX>
void submitTest(queue &Q, range<1> GlobalRange, range<1> LocalRange) {
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<ReqdWGSizeVariadic<Kind, ReqdWGSizeX>>(
        nd_range<1>(GlobalRange, LocalRange),
        [=](nd_item<1>) [[sycl::reqd_work_group_size(ReqdWGSizeX)]] {});
  });
}

template <TestKind Kind, int ReqdWGSizeX, int ReqdWGSizeY, int ReqdWGSizeZ>
void submitTest(queue &Q, range<3> GlobalRange) {
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<
        ReqdWGSizeVariadic<Kind, ReqdWGSizeX, ReqdWGSizeY, ReqdWGSizeZ>>(
        GlobalRange, [=](item<3>) [[sycl::reqd_work_group_size(
                         ReqdWGSizeX, ReqdWGSizeY, ReqdWGSizeZ)]] {});
  });
}

template <TestKind Kind, int ReqdWGSizeX, int ReqdWGSizeY>
void submitTest(queue &Q, range<2> GlobalRange) {
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<ReqdWGSizeVariadic<Kind, ReqdWGSizeX, ReqdWGSizeY>>(
        GlobalRange,
        [=](item<2>)
            [[sycl::reqd_work_group_size(ReqdWGSizeX, ReqdWGSizeY)]] {});
  });
}

template <TestKind Kind, int ReqdWGSizeX>
void submitTest(queue &Q, range<1> GlobalRange) {
  Q.submit([&](handler &CGH) {
    CGH.parallel_for<ReqdWGSizeVariadic<Kind, ReqdWGSizeX>>(
        GlobalRange,
        [=](item<1>) [[sycl::reqd_work_group_size(ReqdWGSizeX)]] {});
  });
}

template <int... ReqdWGSizes>
int runPositiveTest(queue &Q, range<sizeof...(ReqdWGSizes)> GlobalRange,
                    range<sizeof...(ReqdWGSizes)> LocalRange) {
  std::string TestCaseName =
      "Test case ReqdWGSizeVariadic " +
      GetRangeStr(range<sizeof...(ReqdWGSizes)>{ReqdWGSizes...}) + " & " +
      GetRangeStr(LocalRange) + " (Positive with local range)";
  try {
    submitTest<TestKind::PositiveWithLocal, ReqdWGSizes...>(Q, GlobalRange,
                                                            LocalRange);
    Q.wait_and_throw();
    return 0;
  } catch (sycl::exception &E) {
    std::cerr << TestCaseName
              << " failed: unexpected sycl::exception: " << E.what()
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << TestCaseName << " failed: something unexpected has been caught"
              << std::endl;
    return 1;
  }
}

template <int... ReqdWGSizes>
int runPositiveTest(queue &Q, range<sizeof...(ReqdWGSizes)> GlobalRange) {
  std::string TestCaseName =
      "Test case ReqdWGSizeVariadic " +
      GetRangeStr(range<sizeof...(ReqdWGSizes)>{ReqdWGSizes...}) +
      " (Positive without local range)";
  try {
    submitTest<TestKind::PositiveWithoutLocal, ReqdWGSizes...>(Q, GlobalRange);
    Q.wait_and_throw();
    return 0;
  } catch (sycl::exception &E) {
    std::cerr << TestCaseName
              << " failed: unexpected sycl::exception: " << E.what()
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << TestCaseName << " failed: something unexpected has been caught"
              << std::endl;
    return 1;
  }
}

template <int... ReqdWGSizes>
int runNegativeTest(queue &Q, range<sizeof...(ReqdWGSizes)> GlobalRange,
                    range<sizeof...(ReqdWGSizes)> LocalRange) {
  std::string TestCaseName =
      "Test case ReqdWGSizeVariadic " +
      GetRangeStr(range<sizeof...(ReqdWGSizes)>{ReqdWGSizes...}) + " & " +
      GetRangeStr(LocalRange) + " (Negative)";
  try {
    submitTest<TestKind::Negative, ReqdWGSizes...>(Q, GlobalRange, LocalRange);
    Q.wait_and_throw();
    std::cerr << TestCaseName << " failed: no exception has been thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (nd_range_error &E) {
    return 0;
  } catch (sycl::exception &E) {
    std::cerr << TestCaseName
              << " failed: unexpected sycl::exception: " << E.what()
              << std::endl;
    return 1;
  } catch (...) {
    std::cerr << TestCaseName << " failed: something unexpected has been caught"
              << std::endl;
    return 1;
  }
}

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);
  device D(Q.get_device());

  bool IsOpenCL = (D.get_platform().get_backend() == backend::opencl);

  int FailureCounter = 0;

  //****************************************************************************
  // Negative tests with both global and local ranges.
  //****************************************************************************

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4, 4, 4) global, (2, 2, 2) local, reqd_wg_size(2, 2, 2) ->
  // pass
  FailureCounter +=
      runPositiveTest<2, 2, 2>(Q, range<3>(4, 4, 4), range<3>(2, 2, 2));

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4, 4, 4) global, (4, 2, 2) local, reqd_wg_size(4, 2, 2) ->
  // pass
  FailureCounter +=
      runPositiveTest<4, 2, 2>(Q, range<3>(4, 4, 4), range<3>(4, 2, 2));

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4, 4, 4) global, (2, 4, 2) local, reqd_wg_size(2, 4, 2) ->
  // pass
  FailureCounter +=
      runPositiveTest<2, 4, 2>(Q, range<3>(4, 4, 4), range<3>(2, 4, 2));

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4, 4, 4) global, (2, 2, 4) local, reqd_wg_size(2, 2, 4) ->
  // pass
  FailureCounter +=
      runPositiveTest<2, 2, 4>(Q, range<3>(4, 4, 4), range<3>(2, 2, 4));

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4, 4, 4) global, (2, 2, 2) local, reqd_wg_size(2, 2, 2) ->
  // pass
  FailureCounter += runPositiveTest<2, 2>(Q, range<2>(4, 4), range<2>(2, 2));

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4, 4, 4) global, (4, 2, 2) local, reqd_wg_size(4, 2, 2) ->
  // pass
  FailureCounter += runPositiveTest<4, 2>(Q, range<2>(4, 4), range<2>(4, 2));

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4, 4) global, (2, 4) local, reqd_wg_size(2, 4) ->
  // pass
  FailureCounter += runPositiveTest<2, 4>(Q, range<2>(4, 4), range<2>(2, 4));

  // Positive test case: Specify local size that matches required size.
  // parallel_for, (4) global, (2) local, reqd_wg_size(2) ->
  // pass
  FailureCounter += runPositiveTest<2>(Q, range<1>(4), range<1>(2));

  //****************************************************************************
  // Negative tests with only global range.
  //****************************************************************************

  // Kernel that has a required WG size, but no local size is specified.
  //
  // TODO: This fails on OpenCL and should be investigated.
  if (!IsOpenCL) {
    // Positive test case: No local size specified.
    // parallel_for, (8, 8, 8) global, reqd_wg_size(2, 2, 2) -> pass
    FailureCounter += runPositiveTest<2, 2, 2>(Q, range<3>(8, 8, 8));

    // Positive test case: No local size specified.
    // parallel_for, (8, 8, 8) global, reqd_wg_size(4, 2, 2) -> pass
    FailureCounter += runPositiveTest<4, 2, 2>(Q, range<3>(8, 8, 8));

    // Positive test case: No local size specified.
    // parallel_for, (8, 8, 8) global, reqd_wg_size(2, 4, 2) -> pass
    FailureCounter += runPositiveTest<2, 4, 2>(Q, range<3>(8, 8, 8));

    // Positive test case: No local size specified.
    // parallel_for, (8, 8, 8) global, reqd_wg_size(2, 2, 4) -> pass
    FailureCounter += runPositiveTest<2, 2, 4>(Q, range<3>(8, 8, 8));

    // Positive test case: No local size specified.
    // parallel_for, (8, 8) global, reqd_wg_size(2, 2) -> pass
    FailureCounter += runPositiveTest<2, 2>(Q, range<2>(8, 8));

    // Positive test case: No local size specified.
    // parallel_for, (8, 8) global, reqd_wg_size(4, 2) -> pass
    FailureCounter += runPositiveTest<4, 2>(Q, range<2>(8, 8));

    // Positive test case: No local size specified.
    // parallel_for, (8, 8) global, reqd_wg_size(2, 4) -> pass
    FailureCounter += runPositiveTest<2, 4>(Q, range<2>(8, 8));

    // Positive test case: No local size specified.
    // parallel_for, (8) global, reqd_wg_size(2) -> pass
    FailureCounter += runPositiveTest<2>(Q, range<1>(8));
  }

  //****************************************************************************
  // Negative tests where local work-group size has elements that are greater
  // than required in that dimension. reqd_work_group_size is uniform.
  //****************************************************************************

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(2, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 2>(Q, range<3>(8, 8, 8), range<3>(4, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 2) local, reqd_wg_size(2, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 2>(Q, range<3>(8, 8, 8), range<3>(4, 4, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 4, 4) local, reqd_wg_size(2, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 2>(Q, range<3>(8, 8, 8), range<3>(2, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 4, 4) local, reqd_wg_size(2, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 2>(Q, range<3>(8, 8, 8), range<3>(4, 2, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 2, 2) local, reqd_wg_size(2, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 2>(Q, range<3>(8, 8, 8), range<3>(4, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 4, 2) local, reqd_wg_size(2, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 2>(Q, range<3>(8, 8, 8), range<3>(2, 4, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 4) local, reqd_wg_size(2, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 2>(Q, range<3>(8, 8, 8), range<3>(2, 2, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (4, 4) local, reqd_wg_size(2, 2)
  // -> fail
  FailureCounter += runNegativeTest<2, 2>(Q, range<2>(8, 8), range<2>(4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (4, 2) local, reqd_wg_size(2, 2)
  // -> fail
  FailureCounter += runNegativeTest<2, 2>(Q, range<2>(8, 8), range<2>(4, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (2, 4) local, reqd_wg_size(2, 2)
  // -> fail
  FailureCounter += runNegativeTest<2, 2>(Q, range<2>(8, 8), range<2>(2, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8) global, (4) local, reqd_wg_size(2)
  // -> fail
  FailureCounter += runNegativeTest<2>(Q, range<1>(8), range<1>(4));

  //****************************************************************************
  // Negative tests where local work-group size has elements that are less than
  // required in that dimension. reqd_work_group_size is uniform.
  //****************************************************************************

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 2) local, reqd_wg_size(4, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 4>(Q, range<3>(8, 8, 8), range<3>(2, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 4) local, reqd_wg_size(4, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 4>(Q, range<3>(8, 8, 8), range<3>(2, 2, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 2, 2) local, reqd_wg_size(4, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 4>(Q, range<3>(8, 8, 8), range<3>(4, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 2, 2) local, reqd_wg_size(4, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 4>(Q, range<3>(8, 8, 8), range<3>(2, 4, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 4, 4) local, reqd_wg_size(4, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 4>(Q, range<3>(8, 8, 8), range<3>(2, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 2, 4) local, reqd_wg_size(4, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 4>(Q, range<3>(8, 8, 8), range<3>(4, 2, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 2) local, reqd_wg_size(4, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 4>(Q, range<3>(8, 8, 8), range<3>(4, 4, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (2, 2) local, reqd_wg_size(4, 4)
  // -> fail
  FailureCounter += runNegativeTest<4, 4>(Q, range<2>(8, 8), range<2>(2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (2, 4) local, reqd_wg_size(4, 4)
  // -> fail
  FailureCounter += runNegativeTest<4, 4>(Q, range<2>(8, 8), range<2>(2, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (4, 2) local, reqd_wg_size(4, 4)
  // -> fail
  FailureCounter += runNegativeTest<4, 4>(Q, range<2>(8, 8), range<2>(4, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8) global, (2) local, reqd_wg_size(4)
  // -> fail
  FailureCounter += runNegativeTest<4>(Q, range<1>(8), range<1>(2));

  //****************************************************************************
  // Negative tests where local work-group size has elements that are greater
  // than required in that dimension. reqd_work_group_size is non-uniform.
  //****************************************************************************

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 2) local, reqd_wg_size(4, 4, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 2>(Q, range<3>(8, 8, 8), range<3>(2, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 2) local, reqd_wg_size(4, 2, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 2, 4>(Q, range<3>(8, 8, 8), range<3>(2, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 4, 4) local, reqd_wg_size(2, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 4, 4>(Q, range<3>(8, 8, 8), range<3>(2, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 2) local, reqd_wg_size(4, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 2, 2>(Q, range<3>(8, 8, 8), range<3>(2, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 2) local, reqd_wg_size(2, 4, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 4, 2>(Q, range<3>(8, 8, 8), range<3>(2, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (2, 2, 2) local, reqd_wg_size(2, 2, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 4>(Q, range<3>(8, 8, 8), range<3>(2, 2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (2, 2) local, reqd_wg_size(4, 2)
  // -> fail
  FailureCounter += runNegativeTest<4, 2>(Q, range<2>(8, 8), range<2>(2, 2));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (2, 2) local, reqd_wg_size(2, 4)
  // -> fail
  FailureCounter += runNegativeTest<2, 4>(Q, range<2>(8, 8), range<2>(2, 2));

  //****************************************************************************
  // Negative tests where local work-group size has elements that are less than
  // required in that dimension. reqd_work_group_size is non-uniform.
  //****************************************************************************

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(2, 2, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 2, 4>(Q, range<3>(8, 8, 8), range<3>(4, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(4, 2, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 2, 2>(Q, range<3>(8, 8, 8), range<3>(4, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(2, 4, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 4, 2>(Q, range<3>(8, 8, 8), range<3>(4, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(2, 4, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<2, 4, 4>(Q, range<3>(8, 8, 8), range<3>(4, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(4, 2, 4)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 2, 4>(Q, range<3>(8, 8, 8), range<3>(4, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8, 8) global, (4, 4, 2) local, reqd_wg_size(4, 4, 2)
  // -> fail
  FailureCounter +=
      runNegativeTest<4, 4, 2>(Q, range<3>(8, 8, 8), range<3>(4, 4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (4, 4) local, reqd_wg_size(2, 4)
  // -> fail
  FailureCounter += runNegativeTest<2, 4>(Q, range<2>(8, 8), range<2>(4, 4));

  // Negative test case: Specify local size that does not match required size.
  // parallel_for, (8, 8) global, (4, 4) local, reqd_wg_size(4, 2)
  // -> fail
  FailureCounter += runNegativeTest<4, 2>(Q, range<2>(8, 8), range<2>(4, 4));

  return FailureCounter;
}
