///
/// Tests for -Xdevice-post-link
///

// RUN: %clangxx -fsycl --offload-new-driver -Xdevice-post-link "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl --offload-new-driver -Xdevice-post-link=spir64_gen "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'sycl-post-link{{.*}} "foo"'

// RUNx: %clangxx -fsycl --offload-new-driver -fsycl-targets=spir64,spir64_gen -Xdevice-post-link=spir64_gen "foo" -Xdevice-post-link=spir64 "bar" -### %s 2>&1 | \
// RUNx:  FileCheck %s -check-prefix CHECK-MULTIPLE-TARGET --implicit-check-not 'sycl-post-link{{.*}} "foo" "bar"'

// CHECK-SINGLE-TARGET: clang-linker-wrapper{{.*}} {{.*}}--sycl-post-link-options={{.*}}foo{{.*}}

// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xdevice-post-link=spir64_gen foo'

// CHECK-MULTIPLE-TARGET: clang-linker-wrapper{{.*}} {{.*}}--sycl-post-link-options={{.*}}foo{{.*}}bar{{.*}}
