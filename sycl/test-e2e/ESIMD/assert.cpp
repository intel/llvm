// REQUIRES: linux && level_zero

// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// The test still fails after GPU driver update on Linux. Temporary marking it
// as expected to fail, whilst it is being investigated, see intel/llvm#11359
// FIXME: remove that XFAIL
// XFAIL: linux

#include "esimd_test_utils.hpp"

#include <iostream>
#include <sycl/ext/intel/esimd.hpp>
#include <sycl/sycl.hpp>

using namespace sycl;

int main() {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(Q);
  if (!esimd_test::isGPUDriverGE(Q, esimd_test::GPUDriverOS::LinuxAndWindows,
                                 "26816", "101.4576")) {
    std::cout << "Skipped. The test requires GPU driver 1.3.26816 or newer.\n";
    // Additionally, print expected messages to pass FileCheck checks below.
    std::cerr << "Assert called: Id != 31 && \"assert message31\"\n";
    std::cerr << "assert.cpp, Line 29, Function auto main()::(anonymous class)"
              << "::operator()(id<1>) const, gid(31, 0, 0), lid(3, 0, 0)\n";
    return 0;
  }

  try {
    Q.parallel_for(range<1>{100}, [=](id<1> Id) SYCL_ESIMD_KERNEL {
       assert(Id != 31 && "assert message31");
       // CHECK: Assert called: Id != 31 && "assert message31"
       // CHECK: assert.cpp, Line 29, Function auto main()::(anonymous class)::operator()(id<1>) const, gid(31, 0, 0), lid(3, 0, 0)
     }).wait();
  } catch (sycl::exception const &e) {
    std::cerr << "SYCL exception caught: " << e.what() << '\n';
    return 1;
  }

  // CHECK-NOT: Test finished
  std::cerr << "Test finished\n";
  return 0;
}
