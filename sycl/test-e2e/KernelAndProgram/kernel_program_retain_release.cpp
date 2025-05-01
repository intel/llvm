// Tests number of urKernelRetain, urProgramRetain, urKernelRelease,
// urProgramRelease calls made by the SYCL runtime.

// REQUIRES: level_zero

// When in-memory cache is enabled:
//      1. No calls to urKernelRetain and urProgramRetain are made.
//      2. urKernelRelease and urProgramRelease are called once during program shutdown.

// When in-memory cache is disabled:
//      1. No calls to urKernelRetain and urProgramRetain are made.
//      2. urKernelRelease and urProgramRelease are called once after kernel submission.

// When in-memory cache and eviction are both enabled:
//      1. No calls to urKernelRetain and urProgramRetain are made.
//      2. urKernelRelease and urProgramRelease are called once when the item is evicted
//         from the cache or during application shutdown.

// Same kernel is submitted multiple times to the queue.
// RUN: %{build} -DSAME_KERNEL -o %t.out
// RUN: env SYCL_UR_TRACE=-1 %{l0_leak_check} %{run} %t.out &> %t.log
// RUN: FileCheck %s -input-file %t.log

// Different kernel is submitted multiple times to the queue.
// RUN: %{build} -DDIFFERENT_KERNEL -o %t.out
// RUN: env SYCL_UR_TRACE=-1 %{l0_leak_check} %{run} %t.out &> %t1.log
// RUN: FileCheck %s -input-file %t1.log --check-prefix=CHECK-DIFFERENT

// Test submitting kernel bundles.
// RUN: %{build} -DTEST_KERNEL_BUNDLE -o %t.out
// RUN: env SYCL_UR_TRACE=-1 %{l0_leak_check} %{run} %t.out &> %t2.log
// RUN: FileCheck %s -input-file %t2.log --check-prefix=CHECK-KB

// Test submitting free function kernels.
// RUN: %{build} -DTEST_FF_KERNEL -o %t.out
// RUN: env SYCL_UR_TRACE=-1 %{l0_leak_check} %{run} %t.out &> %t3.log
// RUN: FileCheck %s -input-file %t3.log --check-prefix=CHECK-FF

// Test submitting kernel via graph.
// RUN: %{build} -DTEST_GRAPH -o %t.out
// RUN: env SYCL_UR_TRACE=-1 %{l0_leak_check} %{run} %t.out &> %t4.log
// RUN: FileCheck %s -input-file %t4.log --check-prefix=CHECK-GRAPH

#include <sycl/detail/core.hpp>
#include "Inputs/dummy_kernels.hpp"


int main() {
  sycl::queue q;

  // Submit a kernel multiple times.
#ifdef SAME_KERNEL
  // CHECK-COUNT-1: ---> urProgramCreate{{.*}}
  // CHECK-COUNT-1: ---> urKernelCreate{{.*}}
  // CHECK-NOT: ---> urProgramRetain
  // CHECK-NOT: ---> urKernelRetain
  submitSameKernelNTimes(q, 10);
  std::cout << "Initiating program shutdown\n";
  // CHECK-COUNT-1: Initiating program shutdown
  // CHECK-COUNT-1: ---> urKernelRelease
  // CHECK-COUNT-1: ---> urProgramRelease
#endif

// Different kernel is submitted multiple times to the queue.
#ifdef DIFFERENT_KERNEL
  // CHECK-DIFFERENT: ---> urProgramCreate{{.*}}
  // CHECK-DIFFERENT: ---> urKernelCreate
  // CHECK-DIFFERENT: ---> urProgramCreate{{.*}}
  // CHECK-DIFFERENT: ---> urKernelCreate
  // CHECK-DIFFERENT-NOT: ---> urProgramRetain
  // CHECK-DIFFERENT-NOT: ---> urKernelRetain
  submitDifferentKernelNTimes(q, 2);
  std::cout << "Initiating program shutdown\n";
  // CHECK-DIFFERENT-COUNT-1: Initiating program shutdown
  // CHECK-DIFFERENT-COUNT-2: ---> urKernelRelease
  // CHECK-DIFFERENT-COUNT-2: ---> urProgramRelease
#endif

// Test submitting kernel bundles.
#ifdef TEST_KERNEL_BUNDLE
  // CHECK-KB-DAG: ---> urProgramCreateWithIL
  // CHECK-KB-DAG: ---> urProgramLinkExp
  // CHECK-KB-DAG: ---> urKernelCreate
  // CHECK-KB-DAG: ---> urProgramCreateWithIL
  // CHECK-KB-DAG: ---> urKernelCreate

  // CHECK-KB-NOT: ---> urProgramRetain
  // CHECK-KB-NOT: ---> urKernelRetain
  createKernelBundleAndSubmitTwoKernels(q);
  // CHECK-KB-NOT: ---> urKernelRelease

  // For kernel bundles, urProgramRelease is called when kernel bundle
  // and associated device images are released.
  // CHECK-KB-DAG: ---> urProgramRelease
  // CHECK-KB-DAG: ---> urProgramRelease
  std::cout << "Initiating program shutdown\n";
  // CHECK-KB-COUNT-1: Initiating program shutdown
  // CHECK-KB-COUNT-2: ---> urKernelRelease
  // CHECK-KB-COUNT-1: ---> urProgramRelease
#endif

// Test submitting free function kernels.
#ifdef TEST_FF_KERNEL
  // CHECK-FF-DAG: ---> urProgramCreateWithIL
  // CHECK-FF-DAG: ---> urKernelCreate

  // CHECK-FF-NOT: ---> urProgramRetain
  // CHECK-FF-NOT: ---> urKernelRetain
  submitAFreeFunctionKernel(q);
  // CHECK-FF-DAG: ---> urProgramRelease

  std::cout << "Initiating program shutdown\n";
  // CHECK-FF-COUNT-1: Initiating program shutdown
  // CHECK-FF-COUNT-1: ---> urKernelRelease
  // CHECK-FF-COUNT-1: ---> urProgramRelease
#endif

#ifdef TEST_GRAPH
  // CHECK-GRAPH: ---> urProgramCreateWithIL
  // CHECK-GRAPH: ---> urKernelCreate
  // CHECK-GRAPH-NOT: ---> urProgramRetain
  // CHECK-GRAPH-NOT: ---> urKernelRetain
  createAndSubmitAKernelViaGraph(q);

  std::cout << "Initiating program shutdown\n";
  // CHECK-GRAPH-COUNT-1: Initiating program shutdown
  // CHECK-GRAPH-COUNT-1: ---> urKernelRelease
  // CHECK-GRAPH-COUNT-1: ---> urProgramRelease
#endif

  return 0;
}
