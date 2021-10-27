// TODO investigate test failure in CI and re-enable it.
// REQUIRES: TEMPORARILY_DISABLED
// RUN: %clang -fmodules -### %s 2>&1 | FileCheck %s -check-prefix=CHECK-DEFAULT
// CHECK-DEFAULT: -fmodules-cache-path={{.*}}clang{{[/\\]+}}ModuleCache
