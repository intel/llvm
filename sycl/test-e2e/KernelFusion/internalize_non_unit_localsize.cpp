// REQUIRES: fusion
// RUN: %{build} -fsycl-embed-ir -O2 -o %t.out
// RUN: %{run} %t.out

// Test private internalization with "LocalSize" == 3 on buffers that trigger
// special cases in the GEP analyis during remapping.
// - `tmp`:
//   - On SPIR-V and CUDA targets, the IR contains `i8`-typed GEPs to access the
//     elements in the sycl::vec. These GEPs shall _not_ be remapped.
//   - On HIP, the IR contains GEP instructions that add a pointer offset (hence
//     must be remapped) _and_ address into the aggregate element.
// - `tmp2` is an `i8` buffer. The corresponding `i8`-typed GEPs must be
//   remapped during internalization.

#include <sycl/detail/core.hpp>
#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

struct MyStruct {
  int pad;
  sycl::int3 v;
};

int main() {
  constexpr int dataSize = 384;
  std::array<int, dataSize> in, out;
  std::array<MyStruct, dataSize> tmp;
  std::array<char, dataSize> tmp2;

  for (int i = 0; i < dataSize; ++i) {
    in[i] = i;
    tmp[i].v.y() = -1;
    tmp2[i] = -1;
    out[i] = -1;
  }

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  {
    buffer<int> bIn{in.data(), range{dataSize}};
    buffer<MyStruct> bTmp{tmp.data(), range{dataSize}};
    buffer<char> bTmp2{tmp2.data(), range{dataSize}};
    buffer<int> bOut{out.data(), range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn = bIn.get_access(cgh);
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accTmp2 = bTmp2.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      cgh.parallel_for<class KernelOne>(range<1>{dataSize / 3}, [=](id<1> i) {
        accTmp[3 * i].v.y() = accIn[3 * i];
        accTmp[3 * i + 1].v.y() = accIn[3 * i + 1];
        accTmp[3 * i + 2].v.y() = accIn[3 * i + 2];
        accTmp2[3 * i + 2] = static_cast<char>(accIn[3 * i] ^ 0xAA);
        accTmp2[3 * i + 1] = static_cast<char>(accIn[3 * i + 1] ^ 0xAA);
        accTmp2[3 * i] = static_cast<char>(accIn[3 * i + 2] ^ 0xAA);
      });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accTmp2 = bTmp2.get_access(
          cgh, sycl::ext::codeplay::experimental::property::promote_private{});
      auto accIn = bIn.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<class KernelTwo>(range<1>{dataSize / 3}, [=](id<1> i) {
        accOut[3 * i] = accTmp[3 * i].v.y() * accIn[3 * i] + accTmp2[3 * i + 2];
        accOut[3 * i + 1] =
            accTmp[3 * i + 1].v.y() * accIn[3 * i + 1] + accTmp2[3 * i + 1];
        accOut[3 * i + 2] =
            accTmp[3 * i + 2].v.y() * accIn[3 * i + 2] + accTmp2[3 * i];
      });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  for (int i = 0; i < dataSize; ++i) {
    assert(out[i] == (i * i + static_cast<char>(i ^ 0xAA)) &&
           "Computation error");
    assert(tmp[i].v.y() == -1 && tmp2[i] == -1 && "Not internalized");
  }

  return 0;
}
