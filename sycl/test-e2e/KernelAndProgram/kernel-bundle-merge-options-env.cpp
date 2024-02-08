// REQUIRES: gpu
// Disable fallback assert here so, that build process isn't affected
// RUN: %{build} -DSYCL_DISABLE_FALLBACK_ASSERT=1 -o %t.out %debug_option
// RUN: env SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS=-DENV_COMPILE_OPTS SYCL_PROGRAM_LINK_OPTIONS=-DENV_LINK_OPTS SYCL_PROGRAM_APPEND_COMPILE_OPTIONS=-DENV_APPEND_COMPILE_OPTS SYCL_PROGRAM_APPEND_LINK_OPTIONS=-DENV_APPEND_LINK_OPTS %{run} %t.out | FileCheck %s
// Check that options are overrided
// RUN: %{build} -DSYCL_DISABLE_FALLBACK_ASSERT=1 -Xsycl-target-linker=spir64 -DBAR -Xsycl-target-frontend=spir64 -DBAR_COMPILE -o %t.out
// RUN: env SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS=-DENV_COMPILE_OPTS SYCL_PROGRAM_LINK_OPTIONS=-DENV_LINK_OPTS SYCL_PROGRAM_APPEND_COMPILE_OPTIONS=-DENV_APPEND_COMPILE_OPTS SYCL_PROGRAM_APPEND_LINK_OPTIONS=-DENV_APPEND_LINK_OPTS %{run} %t.out | FileCheck %s
// UNSUPPORTED: hip

#include "kernel-bundle-merge-options.hpp"

// CHECK: piProgramBuild
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK: <const char *>:{{[^bar]*}}-DENV_COMPILE_OPTS -DENV_APPEND_COMPILE_OPTS{{[^bar]*}}-DENV_LINK_OPTS -DENV_APPEND_LINK_OPTS{{[^bar]*}}

// CHECK: piProgramCompile(
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK: <const char *>:{{[^bar]*}}-DENV_COMPILE_OPTS -DENV_APPEND_COMPILE_OPTS{{[^bar]*}}

// CHECK: piProgramLink(
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK: <const char *>:{{[^bar]*}}-DENV_LINK_OPTS -DENV_APPEND_LINK_OPTS{{[^bar]*}}
