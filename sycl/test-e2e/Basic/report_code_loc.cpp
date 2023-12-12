/*This test checks that source information where an exception occured is
 * reported*/

// RUN: %{build} -o %t.out
// RUN: env UR_L0_DEBUG=1 %{run} %t.out 2>&1 | FileCheck  %s

#include <sycl/sycl.hpp>
using namespace sycl;
#define XFLOAT float
#define mdlXYZ 1000

int main() {
  bool failed = true;
  XFLOAT *mdlImag;
  queue q{{property::queue::enable_profiling()}};
  mdlImag = sycl::malloc_device<XFLOAT>(mdlXYZ, q);
  try {
    q.memcpy(mdlImag, 0, sizeof(XFLOAT));
  } catch (...) {
    // CHECK: Exception caught at File: {{.*}}report_code_loc.cpp | Function: main | Line: 18 | Column: 5
  }
}
