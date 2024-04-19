// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: env SYCL_PARALLEL_FOR_RANGE_ROUNDING_PARAMS=16:32:64 %{run} %t.out

// Test complete fusion with private internalization specified on the
// accessors for two-dimensional range.

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

template <typename BaseName, size_t sizeX, size_t sizeY> class KernelName;

template <size_t sizeX, size_t sizeY> static void test() {
  constexpr size_t dataSize = sizeX * sizeY;
  int in1[dataSize], in2[dataSize], in3[dataSize], tmp[dataSize], out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp[i] = -1;
    out[i] = -1;
  }

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  {
    range<2> xyRange{sizeY, sizeX};
    buffer<int, 2> bIn1{in1, xyRange};
    buffer<int, 2> bIn2{in2, xyRange};
    buffer<int, 2> bIn3{in3, xyRange};
    buffer<int, 2> bTmp{tmp, xyRange};
    buffer<int, 2> bOut{out, xyRange};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<KernelName<class KernelOne, sizeX, sizeY>>(
          xyRange, [=](id<2> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<KernelName<class KernelTwo, sizeX, sizeY>>(
          xyRange, [=](id<2> i) { accOut[i] = accTmp[i] * accIn3[i]; });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i) && "Computation error");
    assert(tmp[i] == -1 && "Not internalized");
  }
}

int main() {
  // Test power-of-two size.
  test<16, 32>();

  // Test prime sizes large enough to trigger rounded-range kernel insertion.
  // Note that we lower the RR threshold when running this test.
  test<67, 79>();

  return 0;
}
