///
/// Tests for -Xdevice-post-link
///

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xdevice-post-link "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xdevice-post-link=spir64_gen "foo" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-SINGLE-TARGET-UNUSED --implicit-check-not 'sycl-post-link{{.*}} "foo"'

// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -fsycl-targets=spir64,spir64_gen -Xdevice-post-link=spir64_gen "foo" -Xdevice-post-link=spir64 "bar" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TARGET --implicit-check-not 'sycl-post-link{{.*}} "foo" "bar"'

// Each token is forwarded as its own --sycl-post-link-options occurrence.
// RUN: %clangxx -fsycl --offload-new-driver --sysroot=%S/Inputs/SYCL -Xdevice-post-link "foo bar" -### %s 2>&1 | \
// RUN:  FileCheck %s -check-prefix CHECK-MULTIPLE-TOKENS

// CHECK-SINGLE-TARGET: clang-linker-wrapper{{.*}} {{.*}}--sycl-post-link-options={{.*}}foo{{.*}}

// CHECK-SINGLE-TARGET-UNUSED: argument unused during compilation: '-Xdevice-post-link=spir64_gen foo'

// CHECK-MULTIPLE-TARGET: clang-linker-wrapper{{.*}} {{.*}}"--sycl-post-link-options=sycl:spir64-unknown-unknown=bar"{{.*}}"--sycl-post-link-options=sycl:spir64_gen-unknown-unknown=foo"

// CHECK-MULTIPLE-TOKENS: clang-linker-wrapper{{.*}} "--sycl-post-link-options=sycl:spir64-unknown-unknown=foo"{{.*}}"--sycl-post-link-options=sycl:spir64-unknown-unknown=bar"
