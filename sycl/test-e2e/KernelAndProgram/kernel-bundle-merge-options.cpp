// REQUIRES: gpu
// RUN: %{build} -o %t.out %debug_option
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s
// UNSUPPORTED: hip

// Debug option -g is not passed to device code compiler when CL-style driver
// is used and /DEBUG options is passed.
// XFAIL: cl_options

#include "kernel-bundle-merge-options.hpp"

// CHECK: urProgramBuild
// CHECK-SAME: -g

// TODO: Uncomment when build options are properly passed to compile and link
//       commands for kernel_bundle
// xCHECK: urProgramCompile(
// xCHECK-SAME: -g
// xCHECK: urProgramLink(
// xCHECK-SAME: -g
