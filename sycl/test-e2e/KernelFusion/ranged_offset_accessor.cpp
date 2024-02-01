// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: %{run} %t.out

// Test complete fusion with private internalization on accessors with different
// offset and range.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t dataSize = 512;
  int in1[dataSize * 5], in2[dataSize * 5], in3[dataSize * 5],
      tmp[dataSize * 5], out[dataSize * 5];

  size_t offsetIn1 = 0;
  size_t offsetIn2 = 512;
  size_t offsetIn3 = 1024;
  size_t offsetTmp = 1536;
  size_t offsetOut = 2048;

  for (size_t i = 0; i < dataSize; ++i) {
    in1[offsetIn1 + i] = i * 2;
    in2[offsetIn2 + i] = i * 3;
    in3[offsetIn3 + i] = i * 4;
    tmp[offsetTmp + i] = -1;
    out[offsetOut + i] = -1;
  }

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  {
    buffer<int> bIn1{in1, range{dataSize * 5}};
    buffer<int> bIn2{in2, range{dataSize * 5}};
    buffer<int> bIn3{in3, range{dataSize * 5}};
    buffer<int> bTmp{tmp, range{dataSize * 5}};
    buffer<int> bOut{out, range{dataSize * 5}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh, range{516}, id{offsetIn1});
      auto accIn2 = bIn2.get_access(cgh, range{513}, id{offsetIn2});
      auto accTmp = bTmp.get_access(
          cgh, range{514}, id{offsetTmp},
          sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<class KernelOne>(
          dataSize, [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, range{514}, id{offsetTmp},
          sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn3 = bIn3.get_access(cgh, range{515}, id{offsetIn3});
      auto accOut = bOut.get_access(cgh, range{512}, id{offsetOut});
      cgh.parallel_for<class KernelTwo>(
          dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[offsetOut + i] == (20 * i * i) && "Computation error");
    assert(tmp[offsetTmp + i] == -1 && "Not internalized");
  }

  return 0;
}
