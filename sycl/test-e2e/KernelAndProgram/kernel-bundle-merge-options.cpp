// REQUIRES: gpu
// RUN: %{build} -o %t.out -g
// RUN: env SYCL_UR_TRACE=2 %{run} %t.out | FileCheck %s
// UNSUPPORTED: hip

// Rather than %debug_option, we just use -g since it's supported everywhere
// and simplifies things on the FileCheck side.

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
