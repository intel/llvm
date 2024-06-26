// REQUIRES-INTEL-DRIVER: lin: 26816, win: 101.4576
// REQUIRES: linux && level_zero

// RUN: %{build} -DSYCL_FALLBACK_ASSERT=1 -o %t.out
// RUN: %{run} %t.out &> %t.txt ; FileCheck %s --input-file %t.txt
//
// The test still fails after GPU driver update on Linux. Temporary marking it
// as expected to fail, whilst it is being investigated, see intel/llvm#11359
// FIXME: remove that XFAIL
// XFAIL: linux

#include "esimd_test_utils.hpp"

using namespace sycl;

int main() {
  queue Q(esimd_test::ESIMDSelector, esimd_test::createExceptionHandler());
  esimd_test::printTestLabel(Q);

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
