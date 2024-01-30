// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: %{run} %t.out
// XFAIL: hip,cuda

// Test complete fusion with private internalization specified on the
// accessors for a device kernel with sycl::vec::load and sycl::vec::store.

#define VEC 4

#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  constexpr size_t numVec = 512;
  constexpr size_t dataSize = numVec * VEC;
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
      cgh.parallel_for<class KernelOne>(numVec, [=](id<1> i) {
        size_t offset = i;
        vec<int, VEC> in1;
        in1.load(offset,
                 accIn1.template get_multi_ptr<access::decorated::no>());
        vec<int, VEC> in2;
        in2.load(offset,
                 accIn2.template get_multi_ptr<access::decorated::no>());
        auto tmp = in1 + in2;
        tmp.store(offset,
                  accTmp.template get_multi_ptr<sycl::access::decorated::no>());
      });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn3 = bIn3.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(numVec, [=](id<1> i) {
        size_t offset = i;
        vec<int, VEC> tmp;
        tmp.load(offset,
                 accTmp.template get_multi_ptr<sycl::access::decorated::no>());
        vec<int, VEC> in3;
        in3.load(offset,
                 accIn3.template get_multi_ptr<access::decorated::no>());
        auto out = tmp * in3;
        out.store(offset,
                  accOut.template get_multi_ptr<access::decorated::no>());
      });
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

  return 0;
}
