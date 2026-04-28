// REQUIRES: x86-registered-target, symlinks

// RUN: rm -rf %t && mkdir %t
// RUN: ln -s %clang %t/i386-dpclang
// RUN: ln -s %clang %t/x86_64-pc-freebsd13.1-dpclang

// Check if invocation of "foo-dpclang" adds option "-target foo".
//
// RUN: %t/i386-dpclang -c -### %s 2>&1 | FileCheck -check-prefix CHECK-TG1 %s
// CHECK-TG1: Target: i386

// Check if invocation of "foo-clang --target=bar" overrides option "-target foo".
//
// RUN: %t/i386-dpclang -c --target=x86_64 -### %s 2>&1 | FileCheck -check-prefix CHECK-TG2 %s
// CHECK-TG2: Target: x86_64

/// Check if invocation of "arch-vendor-osX.Y-clang" adds option "-target arch-vendor-osX.Y".
// RUN: %t/x86_64-pc-freebsd13.1-dpclang -c -### %s 2>&1 | FileCheck -check-prefix CHECK-TG3 %s
// CHECK-TG3: Target: x86_64-pc-freebsd13.1
