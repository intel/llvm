// TODO: move this to clang/Driver once Native CPU is enabled in CI
// REQUIRES: native_cpu
// TODO: currently disabled on Windows, but could work on Windows if Linux
// libraries are found
// UNSUPPORTED: (windows)
// RUN: %clang -### -fsycl -fsycl-targets=native_cpu -target x86_64-unknown-linux-gnu %s 2> %t.ncpu.out
// RUN: FileCheck %s --input-file %t.ncpu.out
// CHECK: {{(\\|/)}}remangled-l64-signed_char.libspirv-x86_64-unknown-linux-gnu.bc"

// Check that l32 variant is selected for Windows
// RUN: %clang -### -fsycl -fsycl-targets=native_cpu -target x86_64-windows %s 2> %t-win.ncpu.out
// RUN: FileCheck %s --input-file %t-win.ncpu.out --check-prefix=CHECK-WIN
// CHECK-WIN: {{(\\|/)}}remangled-l32-signed_char.libspirv-x86_64-unknown-windows-msvc.bc"
