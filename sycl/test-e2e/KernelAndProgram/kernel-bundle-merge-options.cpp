// REQUIRES: gpu
// RUN: %{build} -o %t.out %debug_option
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s
// UNSUPPORTED: hip

// Depends on SPIR-V Backend & run-time drivers version.
// XFAIL: spirv-backend && gpu
// XFAIL-TRACKER: CMPLRLLVM-64705

// Note that the UR call might be urProgramBuild OR urProgramBuildExp .
// The same is true for Compile and Link.
// We want the first match. Don't put parentheses after.

#include "kernel-bundle-merge-options.hpp"

// CHECK: <--- urProgramBuild
// CHECK-SAME: -g

// CHECK: <--- urProgramCompile
// CHECK-SAME: -g

// TODO: Uncomment when build options are properly passed to link
//       commands for kernel_bundle
// xCHECK: <--- urProgramLink
// xCHECK-SAME: -g
