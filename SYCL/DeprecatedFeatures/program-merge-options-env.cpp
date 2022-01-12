// RUN: %clangxx -fsycl -D__SYCL_INTERNAL_API %s -o %t.out %debug_option
// RUN: env SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS=-DENV_COMPILE_OPTS SYCL_PROGRAM_LINK_OPTIONS=-DENV_LINK_OPTS SYCL_DEVICE_FILTER=%sycl_be %t.out | FileCheck %s
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// UNSUPPORTED: ze_debug-1,ze_debug4
#include "program-merge-options.hpp"

// CHECK: piProgramBuild
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <const char *>:{{.*}}-DENV_COMPILE_OPTS{{.*}}-vc-codegen

// TODO: Uncomment when build options are properly passed to compile and link
//       commands for program
// xCHECK: piProgramCompile(
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <const char *>: -DENV_COMPILE_OPTS -vc-codegen
// xCHECK: piProgramLink(
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <unknown>
// xCHECK-NEXT: <const char *>: -DENV_LINK_OPTS -vc-codegen
