// Disable fallback assert here so, that build process isn't affected
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%sycl_triple %s -o %t.out %debug_option
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS=-DENV_COMPILE_OPTS SYCL_PROGRAM_LINK_OPTIONS=-DENV_LINK_OPTS %t.out %GPU_CHECK_PLACEHOLDER
// Check that options are overrided
// RUN: %clangxx -DSYCL_DISABLE_FALLBACK_ASSERT=1 -fsycl -fsycl-targets=%sycl_triple -Xsycl-target-linker=spir64 -bar -Xsycl-target-frontend=spir64 -bar_compile %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER SYCL_PI_TRACE=-1 SYCL_PROGRAM_COMPILE_OPTIONS=-DENV_COMPILE_OPTS SYCL_PROGRAM_LINK_OPTIONS=-DENV_LINK_OPTS %t.out %GPU_CHECK_PLACEHOLDER
// REQUIRES: gpu
// UNSUPPORTED: hip

#include "kernel-bundle-merge-options.hpp"

// CHECK: piProgramBuild
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK: <const char *>:{{[^bar]*}}-DENV_COMPILE_OPTS{{[^bar]*}}-DENV_LINK_OPTS{{[^bar]*}}

// CHECK: piProgramCompile(
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK: <const char *>:{{[^bar]*}}-DENV_COMPILE_OPTS{{[^bar]*}}

// CHECK: piProgramLink(
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK-NEXT: <unknown>
// CHECK: <const char *>:{{[^bar]*}}-DENV_LINK_OPTS{{[^bar]*}}
