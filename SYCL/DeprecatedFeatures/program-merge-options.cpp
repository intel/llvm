// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out %debug_option
// RUN: env SYCL_PI_TRACE=-1 SYCL_DEVICE_FILTER=%sycl_be %t.out | FileCheck %s
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// UNSUPPORTED: ze_debug-1,ze_debug4
// Debug option -g is not passed to device code compiler when CL-style driver
// is used and /DEBUG options is passed.
// XFAIL: cl_options
#include "program-merge-options.hpp"

// CHECK: piProgramBuild
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <const char *>:{{.*}}-DBUILD_OPTS{{.*}}-g{{.*}}-vc-codegen

// TODO: Uncomment when build options are properly passed to compile and link
//       commands for program
// xCHECK: piProgramCompile(
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <const char *>: -DCOMPILE_OPTS -vc-codegen
// xCHECK: piProgramLink(
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <const char *>: -cl-fast-relaxed-math -vc-codegen
