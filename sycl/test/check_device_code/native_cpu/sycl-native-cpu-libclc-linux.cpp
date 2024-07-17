// TODO: move this to clang/Driver once Native CPU is enabled in CI
// REQUIRES: native_cpu && linux
// RUN: %clang -### -fsycl -fsycl-targets=native_cpu -target x86_64-unknown-linux-gnu %s 2> %t.ncpu.out
// RUN: FileCheck %s --input-file %t.ncpu.out
// CHECK: {{(\\|/)}}remangled-l64-signed_char.libspirv-x86_64-unknown-linux-gnu.bc"
