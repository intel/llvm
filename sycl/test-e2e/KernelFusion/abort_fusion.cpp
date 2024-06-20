// RUN: %{build} %{embed-ir} -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %{run} %t.out 2>&1 | FileCheck %s

// Test fusion being aborted: Different scenarios causing the JIT compiler
// to abort fusion due to constraint violations for fusion. Also check that
// warnings are printed when SYCL_RT_WARNING_LEVEL=1.

#include <sycl/detail/core.hpp>

#include <sycl/ext/codeplay/experimental/fusion_wrapper.hpp>
#include <sycl/properties/all_properties.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

enum class Internalization { None, Local, Private };

template <typename Range> size_t getSize(Range r);

template <int Dimensions> size_t getSize(range<Dimensions> r) {
  return r.size();
}
template <int Dimensions> size_t getSize(nd_range<Dimensions> r) {
  return r.get_global_range().size();
}

template <int N> auto global_linear_id(sycl::nd_item<N> ndi) {
  return ndi.get_global_linear_id();
}
template <int N> auto global_linear_id(sycl::item<N> i) {
  return i.get_linear_id();
}

template <typename Kernel1Name, typename Kernel2Name, typename Range1,
          typename Range2>
void performFusion(queue &q, Range1 R1, Range2 R2) {
  int in[dataSize], tmp[dataSize], out[dataSize];

  for (size_t i = 0; i < dataSize; ++i) {
    in[i] = i;
    tmp[i] = -1;
    out[i] = -1;
  }
  {
    buffer<int> bIn{in, range{dataSize}};
    buffer<int> bTmp{tmp, range{dataSize}};
    buffer<int> bOut{out, range{dataSize}};

    ext::codeplay::experimental::fusion_wrapper fw(q);
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn = bIn.get_access(cgh);
      auto accTmp = bTmp.get_access(cgh);
      cgh.parallel_for<Kernel1Name>(R1, [=](auto i) {
        size_t j = global_linear_id(i);
        accTmp[j] = accIn[j] + 5;
      });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<Kernel2Name>(R2, [=](auto i) {
        size_t j = global_linear_id(i);
        accOut[j] = accTmp[j] * 2;
      });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  size_t numErrors = 0;
  size_t size = getSize(R1);
  for (size_t i = 0; i < size; ++i) {
    if (out[i] != ((i + 5) * 2)) {
      ++numErrors;
    }
  }
  if (numErrors) {
    std::cout << "COMPUTATION ERROR\n";
  } else {
    std::cout << "COMPUTATION OK\n";
  }

  assert(numErrors == 0);
}

static void emptyFusionList(queue &q) {
  ext::codeplay::experimental::fusion_wrapper fw(q);
  fw.start_fusion();
  assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");
  fw.complete_fusion();
  assert(!fw.is_in_fusion_mode() &&
         "Queue should not be in fusion mode anymore");
}

int main() {

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  // Scenario: Fusing two kernels with different local size should lead to
  // fusion being aborted.
  performFusion<class Kernel1_3, class Kernel2_3>(
      q, nd_range<1>{range<1>{dataSize}, range<1>{16}},
      nd_range<1>{range<1>{dataSize}, range<1>{8}});
  // CHECK: ERROR: JIT compilation for kernel fusion failed with message:
  // CHECK-NEXT: Illegal ND-range combination
  // CHECK-NEXT: Detailed information:
  // CHECK-NEXT: Cannot fuse kernels with different local sizes
  // CHECK: COMPUTATION OK

  // Scenario: An empty fusion list should not be classified as having
  // incompatible ND ranges.
  emptyFusionList(q);
  // CHECK-NOT: Illegal ND-range combination
  // CHECK: WARNING: Fusion list is empty

  // Scenario: Fusing two kernels that would lead to non-uniform work-group
  // sizes should lead to fusion being aborted.
  performFusion<class Kernel1_4, class Kernel2_4>(
      q, nd_range<1>{range<1>{9}, range<1>{3}}, range<1>{dataSize});
  // CHECK: ERROR: JIT compilation for kernel fusion failed with message:
  // CHECK-NEXT: Illegal ND-range combination
  // CHECK-NEXT: Detailed information:
  // CHECK-NEXT: Cannot fuse kernels whose fusion would yield non-uniform work-group sizes
  // CHECK: COMPUTATION OK

  // Scenario: Fusing two kernels that may lead to synchronization issues as two
  // work-items in the same work-group may not be in the same work-group in the
  // fused ND-range.
  performFusion<class Kernel1_5, class Kernel2_5>(
      q, nd_range<2>{range<2>{2, 2}, range<2>{2, 2}},
      nd_range<2>{range<2>{4, 4}, range<2>{2, 2}});
  // CHECK: ERROR: JIT compilation for kernel fusion failed with message:
  // CHECK-NEXT: Illegal ND-range combination
  // CHECK-NEXT: Detailed information:
  // CHECK-NEXT: Cannot fuse kernels when any of the fused kernels with a specific local size has different global sizes in dimensions [2, N) or different number of dimensions
  // CHECK: COMPUTATION OK

  return 0;
}
