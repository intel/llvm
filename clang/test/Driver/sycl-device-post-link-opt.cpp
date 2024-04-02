///
/// Tests for -Xdevice-post-link
///

// RUN: %clangxx -fsycl -Xdevice-post-link "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl -Xdevice-post-link=spir64_gen "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'sycl-post-link{{.*}} "foo"'

// RUN: %clangxx -fsycl -fsycl-targets=spir64,spir64_gen -Xdevice-post-link=spir64_gen "foo" -Xdevice-post-link=spir64 "bar" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TARGET --implicit-check-not 'sycl-post-link{{.*}} "foo" "bar"'

// CHECK-SINGLE-TARGET: sycl-post-link{{.*}} "foo"

// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xdevice-post-link=spir64_gen foo'

// CHECK-MULTIPLE-TARGET: sycl-post-link{{.*}} "bar"
// CHECK-MULTIPLE-TARGET: clang-offload-wrapper{{.*}} "-target=spir64" "-kind=sycl"
// CHECK-MULTIPLE-TARGET: sycl-post-link{{.*}} "foo"
// CHECK-MULTIPLE-TARGET: clang-offload-wrapper{{.*}} "-target=spir64_gen" "-kind=sycl"
