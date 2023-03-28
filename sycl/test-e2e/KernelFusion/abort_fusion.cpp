// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: env SYCL_RT_WARNING_LEVEL=1 %CPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %CPU_CHECK_PLACEHOLDER
// RUN: env SYCL_RT_WARNING_LEVEL=1 %GPU_RUN_PLACEHOLDER %t.out 2>&1\
// RUN: %GPU_CHECK_PLACEHOLDER
// UNSUPPORTED: cuda || hip
// REQUIRES: fusion

// Test fusion being aborted: Different scenarios causing the JIT compiler
// to abort fusion due to constraint violations for fusion. Also check that
// warnings are printed when SYCL_RT_WARNING_LEVEL=1.

#include <sycl/sycl.hpp>

using namespace sycl;

constexpr size_t dataSize = 512;

enum class Internalization { None, Local, Private };

template <typename Kernel1Name, typename Kernel2Name, int Kernel1Dim>
void performFusion(queue &q, range<Kernel1Dim> k1Global,
                   range<Kernel1Dim> k1Local) {
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
      cgh.parallel_for<Kernel1Name>(nd_range<Kernel1Dim>{k1Global, k1Local},
                                    [=](item<Kernel1Dim> i) {
                                      auto LID = i.get_linear_id();
                                      accTmp[LID] = accIn[LID] + 5;
                                    });
    });

    q.submit([&](handler &cgh) {
      auto accTmp = bTmp.get_access(cgh);
      auto accOut = bOut.get_access(cgh);
      cgh.parallel_for<Kernel2Name>(nd_range<1>{{dataSize}, {8}}, [=](id<1> i) {
        accOut[i] = accTmp[i] * 2;
      });
    });

    fw.complete_fusion({ext::codeplay::experimental::property::no_barriers{}});

    assert(!fw.is_in_fusion_mode() &&
           "Queue should not be in fusion mode anymore");
  }

  // Check the results
  size_t numErrors = 0;
  for (size_t i = 0; i < k1Global.size(); ++i) {
    if (out[i] != ((i + 5) * 2)) {
      ++numErrors;
    }
  }
  if (numErrors) {
    std::cout << "COMPUTATION ERROR\n";
  } else {
    std::cout << "COMPUTATION OK\n";
  }
}

int main() {

  queue q{ext::codeplay::experimental::property::queue::enable_fusion{}};

  // Scenario: Fusing two kernels with different local size should lead to
  // fusion being aborted.
  performFusion<class Kernel1_3, class Kernel2_3>(q, range<1>{dataSize},
                                                  range<1>{16});
  // CHECK:      ERROR: JIT compilation for kernel fusion failed with message:
  // CHECK-NEXT: Cannot fuse kernels with different offsets or local sizes
  // CHECK-NEXT: COMPUTATION OK

  return 0;
}
