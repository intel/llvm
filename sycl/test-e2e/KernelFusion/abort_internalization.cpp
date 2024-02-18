// RUN: %{build} -O2 -fsycl-embed-ir -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 SYCL_ENABLE_FUSION_CACHING=0 %{run} %t.out 2>&1 | FileCheck %s

// Test incomplete internalization: Different scenarios causing the JIT compiler
// to abort internalization due to target or parameter mismatch. Also check that
// warnings are printed when SYCL_RT_WARNING_LEVEL=1.

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

enum class Internalization { None, Local, Private };

void performFusion(queue &q, Internalization intKernel1,
                   size_t localSizeKernel1, Internalization intKernel2,
                   size_t localSizeKernel2,
                   bool expectInternalization = false) {
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

    ext::codeplay::experimental::fusion_wrapper fw{q};
    fw.start_fusion();

    assert(fw.is_in_fusion_mode() && "Queue should be in fusion mode");

    q.submit([&](handler &cgh) {
      auto accIn = bIn.get_access(cgh);
      property_list properties{};
      if (intKernel1 == Internalization::Private) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_private{}};
      } else if (intKernel1 == Internalization::Local) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_local{}};
      }
      accessor<int> accTmp = bTmp.get_access(cgh, properties);

      if (localSizeKernel1 > 0) {
        cgh.parallel_for<class Kernel1>(
            nd_range<1>{{dataSize}, {localSizeKernel1}},
            [=](id<1> i) { accTmp[i] = accIn[i] + 5; });
      } else {
        cgh.parallel_for<class KernelOne>(
            dataSize, [=](id<1> i) { accTmp[i] = accIn[i] + 5; });
      }
    });

    q.submit([&](handler &cgh) {
      property_list properties{};
      if (intKernel2 == Internalization::Private) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_private{}};
      } else if (intKernel2 == Internalization::Local) {
        properties = {
            sycl::ext::codeplay::experimental::property::promote_local{}};
      }
      accessor<int> accTmp = bTmp.get_access(cgh, properties);
      auto accOut = bOut.get_access(cgh);
      if (localSizeKernel2 > 0) {
        cgh.parallel_for<class Kernel2>(
            nd_range<1>{{dataSize}, {localSizeKernel2}},
            [=](id<1> i) { accOut[i] = accTmp[i] * 2; });
      } else {
        cgh.parallel_for<class KernelTwo>(
            dataSize, [=](id<1> i) { accOut[i] = accTmp[i] * 2; });
      }
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  size_t numErrors = 0;
  size_t numInternalized = 0;
  for (size_t i = 0; i < dataSize; ++i) {
    if (out[i] != ((i + 5) * 2)) {
      ++numErrors;
    }
    if (tmp[i] == -1) {
      ++numInternalized;
    }
  }
  if (numErrors) {
    std::cout << "COMPUTATION ERROR\n";
    return;
  }
  if (!expectInternalization && numInternalized) {
    std::cout << "WRONG INTERNALIZATION\n";
    return;
  }
  std::cout << "COMPUTATION OK\n";
}

int main() {
  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  // Scenario: One accessor without internalization, one with local
  // internalization. Should fall back to no internalization and print a
  // warning.
  std::cout << "None, Local(0)\n";
  performFusion(q, Internalization::None, 0, Internalization::Local, 0);
  // CHECK: None, Local(0)
  // CHECK: WARNING: Not performing specified local promotion, due to previous mismatch or because previous accessor specified no promotion
  // CHECK: COMPUTATION OK

  // Scenario: One accessor without internalization, one with private
  // internalization. Should fall back to no internalization and print a
  // warning.
  std::cout << "None, Private\n";
  performFusion(q, Internalization::None, 0, Internalization::Private, 0);
  // CHECK: None, Private
  // CHECK: WARNING: Not performing specified private promotion, due to previous mismatch or because previous accessor specified no promotion
  // CHECK: COMPUTATION OK

  // Scenario: Both accessor with local promotion, but the second kernel does
  // not specify a work-group size. No promotion should happen and a warning
  // should be printed.
  std::cout << "Local(8), Local(0)\n";
  performFusion(q, Internalization::Local, 8, Internalization::Local, 0);
  // CHECK: Local(8), Local(0)
  // CHECK: WARNING: Work-group size for local promotion not specified, not performing internalization
  // CHECK: COMPUTATION OK

  // Scenario: Both accessor with local promotion, but the first kernel does
  // not specify a work-group size. No promotion should happen and a warning
  // should be printed.
  std::cout << "Local(0), Local(8)\n";
  performFusion(q, Internalization::Local, 0, Internalization::Local, 8);
  // CHECK: Local(0), Local(8)
  // CHECK: WARNING: Work-group size for local promotion not specified, not performing internalization
  // CHECK: WARNING: Not performing specified local promotion, due to previous mismatch or because previous accessor specified no promotion
  // CHECK: COMPUTATION OK

  // Scenario: Both accessor with local promotion, but the kernels specify
  // different work-group sizes. No promotion should happen and a warning should
  // be printed.
  std::cout << "Local(8), Local(16)\n";
  performFusion(q, Internalization::Local, 8, Internalization::Local, 16);
  // CHECK: Local(8), Local(16)
  // CHECK: WARNING: Not performing specified local promotion due to work-group size mismatch
  // CHECK: ERROR: JIT compilation for kernel fusion failed with message:
  // CHECK: Illegal ND-range combination
  // CHECK: Detailed information:
  // CHECK: Cannot fuse kernels with different local sizes
  // CHECK: COMPUTATION OK

  // Scenario: One accessor with local internalization, one with private
  // internalization. Should fall back to local internalization and print a
  // warning.
  std::cout << "Local(8), Private(8)\n";
  performFusion(q, Internalization::Local, 8, Internalization::Private, 8,
                /* expectInternalization */ true);
  // CHECK: Local(8), Private(8)
  // CHECK: WARNING: Performing local internalization instead, because previous accessor specified local promotion
  // CHECK: COMPUTATION OK

  return 0;
}
