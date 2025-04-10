// REQUIRES: gpu
// RUN: %{build} -o %t.out %debug_option
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s
// UNSUPPORTED: hip

#include "kernel-bundle-merge-options.hpp"

// CHECK: <--- urProgramBuild(
// CHECK-SAME: -g

// CHECK: <--- urProgramCompile(
// CHECK-SAME: -g

// TODO: Uncomment when build options are properly passed to link
//       commands for kernel_bundle
// xCHECK: <--- urProgramLink(
// xCHECK-SAME: -g
