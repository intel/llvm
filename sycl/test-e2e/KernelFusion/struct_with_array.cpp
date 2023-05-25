// REQUIRES: fusion
// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: %{run} %t.out

// Test complete fusion with private internalization on a kernel functor with an
// array member.

#include <sycl/sycl.hpp>

using namespace sycl;

struct KernelTwo {
  accessor<int> buf;
  accessor<int> out;
  int coef[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  KernelTwo(accessor<int> buf, accessor<int> out) : buf{buf}, out{out} {}

  void operator()(nd_item<1> i) const {
    out[i.get_global_linear_id()] =
        buf[i.get_global_linear_id()] * coef[i.get_local_linear_id()];
  }
};

int main() {
  constexpr size_t dataSize = 512;
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
    buffer<int> bIn1{in1, range{dataSize}};
    buffer<int> bIn2{in2, range{dataSize}};
    buffer<int> bIn3{in3, range{dataSize}};
    buffer<int> bTmp{tmp, range{dataSize}};
    buffer<int> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<class KernelOne>(
          nd_range<1>{{dataSize}, {8}},
          [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for(nd_range<1>{{dataSize}, {8}}, KernelTwo{accTmp, accOut});
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (size_t i = 0; i < dataSize; ++i) {
    assert(out[i] == (5 * i * (i % 8)) && "Computation error");
    assert(tmp[i] == -1 && "Not internalized");
  }

  return 0;
}
