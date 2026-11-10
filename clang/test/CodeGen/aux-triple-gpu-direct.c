// Check that -aux-triple sets the ABI of a direct (non-offloading) NVPTX or
// SPIR-V compilation to the host's data model.

// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,LP64
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,LP64
// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda \
// RUN:   -aux-triple x86_64-pc-windows-msvc -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,LLP64
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,LP64
// RUN: %clang_cc1 -triple spirv64-unknown-unknown \
// RUN:   -aux-triple x86_64-pc-windows-msvc -emit-llvm -o - %s \
// RUN:   | FileCheck %s --check-prefixes=CHECK,LLP64

typedef struct {
  long quot;
  long rem;
} ldiv_t;

// LP64: @ldiv_size = {{.*}}global i32 16
// LLP64: @ldiv_size = {{.*}}global i32 8
int ldiv_size = sizeof(ldiv_t);

// LP64: define {{.*}}double @scalbln(double {{.*}}, i64 {{.*}})
// LLP64: define {{.*}}double @scalbln(double {{.*}}, i32 {{.*}})
double scalbln(double x, long n) { return x * n; }

// CHECK: define {{.*}}i64 @llabs(i64
long long llabs(long long n) { return n < 0 ? -n : n; }
