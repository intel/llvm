// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: %{run} %t.out

// Test complete fusion with local internalization specified on the
// accessors for a combination of four kernels, forming a diamond-like shape and
// repeating one of the kernels.

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

struct AddKernel {
  accessor<int, 1> accIn1;
  accessor<int, 1> accIn2;
  accessor<int, 1> accOut;

  void operator()(id<1> i) const { accOut[i] = accIn1[i] + accIn2[i]; }
};

int main() {
  constexpr size_t dataSize = 512;
  int in1[dataSize], in2[dataSize], in3[dataSize], tmp1[dataSize],
      tmp2[dataSize], tmp3[dataSize], out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i] = i * 2;
    in2[i] = i * 3;
    in3[i] = i * 4;
    tmp1[i] = -1;
    tmp2[i] = -1;
    tmp3[i] = -1;
    out[i] = -1;
  }

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  {
    buffer<int> bIn1{in1, range{dataSize}};
    buffer<int> bIn2{in2, range{dataSize}};
    buffer<int> bIn3{in3, range{dataSize}};
    buffer<int> bTmp1{
        tmp1,
        range{dataSize},
        {sycl::ext::codeplay::experimental::property::promote_local{}}};
    buffer<int> bTmp2{
        tmp2,
        range{dataSize},
        {sycl::ext::codeplay::experimental::property::promote_local{}}};
    buffer<int> bTmp3{
        tmp3,
        range{dataSize},
        {sycl::ext::codeplay::experimental::property::promote_local{}}};
    buffer<int> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp1 = bTmp1.get_access(cgh);
      cgh.parallel_for<AddKernel>(nd_range<1>{{dataSize}, {16}},
                                  AddKernel{accIn1, accIn2, accTmp1});
    });

    q.submit([&](handler &cgh) {
      auto accTmp1 = bTmp1.get_access(cgh);
      auto accIn3 = bIn3.get_access(cgh);
      auto accTmp2 = bTmp2.get_access(cgh);
      cgh.parallel_for<class KernelOne>(
          nd_range<1>{{dataSize}, {16}},
          [=](id<1> i) { accTmp2[i] = accTmp1[i] * accIn3[i]; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp1 = bTmp1.get_access(cgh);
      auto accTmp3 = bTmp3.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          nd_range<1>{{dataSize}, {16}},
          [=](id<1> i) { accTmp3[i] = accTmp1[i] * 5; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp2 = bTmp2.get_access(cgh);
      auto accTmp3 = bTmp3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<AddKernel>(nd_range<1>{{dataSize}, {16}},
                                  AddKernel{accTmp2, accTmp3, accOut});
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (20 * i * i + i * 25) && "Computation error");
    assert(tmp1[i] == -1 && "tmp1 not internalized");
    assert(tmp2[i] == -1 && "tmp2 not internalized");
    assert(tmp3[i] == -1 && "tmp3 not internalized");
  }

  return 0;
}
