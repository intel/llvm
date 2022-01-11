// Disable fallback assert here so, that build process isn't affected
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT=1 -fsycl %s -o %t.out %debug_option
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS=-DENV_COMPILE_OPTS SYCL_PROGRAM_LINK_OPTIONS=-DENV_LINK_OPTS %t.out %GPU_CHECK_PLACEHOLDER
// REQUIRES: gpu
// UNSUPPORTED: cuda || hip
// UNSUPPORTED: ze_debug-1,ze_debug4
#include "kernel-bundle-merge-options.hpp"

// CHECK: piProgramBuild
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <const char *>:{{.*}}-DENV_COMPILE_OPTS{{.*}}-vc-codegen

// TODO: Uncomment when build options are properly passed to compile and link
//       commands for kernel_bundle
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
