// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: %{run} %t.out

// Test complete fusion with internalization of a struct type.

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t dataSize = 512;

  vec<int, 4> in1[dataSize], in2[dataSize], in3[dataSize], tmp[dataSize],
      out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in1[i].s0() = in1[i].s1() = in1[i].s2() = in1[i].s3() = i * 2;
    in2[i].s0() = in2[i].s1() = in2[i].s2() = in2[i].s3() = i * 3;
    in3[i].s0() = in3[i].s1() = in3[i].s2() = in3[i].s3() = i * 4;
    tmp[i].s0() = tmp[i].s1() = tmp[i].s2() = tmp[i].s3() = -1;
    out[i].s0() = out[i].s1() = out[i].s2() = out[i].s3() = -1;
  }

  queue q{default_selector_v,
          {ext::codeplay::experimental::property::queue::enable_fusion{}}};

  {
    buffer<vec<int, 4>> bIn1{in1, range{dataSize}};
    buffer<vec<int, 4>> bIn2{in2, range{dataSize}};
    buffer<vec<int, 4>> bIn3{in3, range{dataSize}};
    buffer<vec<int, 4>> bTmp{tmp, range{dataSize}};
    buffer<vec<int, 4>> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn1 = bIn1.get_access(cgh);
      auto accIn2 = bIn2.get_access(cgh);
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<class KernelOne>(
          dataSize, [=](id<1> i) { accTmp[i] = accIn1[i] + accIn2[i]; });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(
          dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * accIn3[i]; });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  constexpr vec<int, 4> not_written{-1, -1, -1, -1};
  for (size_t i = 0; i < dataSize; ++i) {
    const vec<int, 4> expected{20 * i * i, 20 * i * i, 20 * i * i, 20 * i * i};
    assert(all(out[i] == expected) && "Computation error");
    assert(all(tmp[i] == not_written) && "Not internalizing");
  };

  return 0;
}
