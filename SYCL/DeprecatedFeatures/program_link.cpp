// RUN: %clangxx -D__SYCL_INTERNAL_API -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_LINK_OPTIONS="-cl-fast-relaxed-math" %t.out %CPU_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-RELAXED-MATH
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_LINK_OPTIONS="-cl-fast-relaxed-math" %t.out %GPU_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-RELAXED-MATH
// RUN: %ACC_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_LINK_OPTIONS="-cl-fast-relaxed-math" %t.out %ACC_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-RELAXED-MATH

// RUN: %CPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS="-cl-opt-disable" %t.out %CPU_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-OPT-DISABLE
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS="-cl-opt-disable" %t.out %GPU_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-OPT-DISABLE
// RUN: %ACC_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS="-cl-opt-disable" %t.out %ACCPU_CHECK_PLACEHOLDER --check-prefix=CHECK-IS-OPT-DISABLE
//
// Hits an assertion on AMD with multiple GPUs available, fails trace on Nvidia.
// XFAIL: hip_amd || hip_nvidia
//
// Unsupported on Level Zero because the test passes OpenCL specific compiler
// and linker switches.
// UNSUPPORTED: level_zero

#include <CL/sycl.hpp>
#include <iostream>
using namespace cl::sycl;
class DUMMY {
public:
  void operator()(item<1>) const {};
};

int main(void) {
  default_selector s;
  platform p(s);
  if (p.is_host()) {
    return 0;
  }
  context c(p);
  queue Q(c, s);
  program prog1(c);
  prog1.compile_with_kernel_type<DUMMY>();
  prog1.link("-cl-finite-math-only");
  assert(prog1.get_state() == cl::sycl::program_state::linked &&
         "fail to link program");
  // CHECK-IS-RELAXED-MATH: -cl-fast-relaxed-math
  // CHECK-IS-RELAXED-MATH-NOT: -cl-finite-math-only
  assert(prog1.get_link_options() == "-cl-finite-math-only" &&
         "program::get_link_options() output is wrong");

  program prog2(c);
  prog2.compile_with_kernel_type<DUMMY>("-cl-mad-enable");
  assert(prog2.get_state() == cl::sycl::program_state::compiled &&
         "fail to compile program");
  // CHECK-IS-OPT-DISABLE: -cl-opt-disable
  // CHECK-IS-OPT-DISABLE-NOT: -cl-mad-enable
  assert(prog2.get_compile_options() == "-cl-mad-enable" &&
         "program::get_compile_options() output is wrong");

  // enforce SYCL toolchain to emit device image but no enqueue in run-time
  if (false) {
    Q.submit([&](handler &CGH) { CGH.parallel_for(range<1>{2}, DUMMY{}); });
  }

  return 0;
}
