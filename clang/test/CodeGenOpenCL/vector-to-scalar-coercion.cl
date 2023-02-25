// RUN: %clang_cc1 -x cl -triple i686-pc-win32-gnu    -fopencl-force-vector-abi %s -O0 -emit-llvm -o - | FileCheck %s --check-prefix NOCOER
// RUN: %clang_cc1 -x cl -triple x86_64-unknown-linux -fopencl-force-vector-abi %s -O0 -emit-llvm -o - | FileCheck %s --check-prefix NOCOER
// RUN: %clang_cc1 -x cl -triple x86_64-pc-win32-gnu  -fopencl-force-vector-abi %s -O0 -emit-llvm -o - | FileCheck %s --check-prefix NOCOER

// RUN: %clang_cc1 -x cl -triple i686-pc-win32-gnu %s    -O0 -emit-llvm -o - | FileCheck %s --check-prefix COER32CL
// RUN: %clang_cc1 -x cl -triple x86_64-unknown-linux %s -O0 -emit-llvm -o - | FileCheck %s --check-prefix COER64
// RUN: %clang_cc1 -x cl -triple x86_64-pc-win32-gnu %s  -O0 -emit-llvm -o - | FileCheck %s --check-prefix NOCOER

// RUN: %clang_cc1 -x c  -triple i686-pc-win32-gnu %s    -O0 -emit-llvm -o - | FileCheck %s --check-prefix COER32
// RUN: %clang_cc1 -x c  -triple x86_64-unknown-linux %s -O0 -emit-llvm -o - | FileCheck %s --check-prefix COER64
// RUN: %clang_cc1 -x c  -triple x86_64-pc-win32-gnu %s  -O0 -emit-llvm -o - | FileCheck %s --check-prefix NOCOER-C-WIN

typedef unsigned short ushort;
typedef ushort ushort4 __attribute__((ext_vector_type(4)));

typedef unsigned long ulong;
typedef ulong ulong4 __attribute__((ext_vector_type(4)));

ulong4 __attribute__((const)) __attribute__((overloadable)) convert_ulong4_rte(ushort4 x)
{
  return 1;
}

// NOCOER:       define {{.*}}<4 x i64> @_Z18convert_ulong4_rteDv4_t(<4 x i16> noundef %{{.*}})
// NOCOER-C-WIN: define {{.*}}<4 x i32> @_Z18convert_ulong4_rteDv4_t(<4 x i16> noundef %{{.*}})
// COER32CL:     define {{.*}}<4 x i64> @_Z18convert_ulong4_rteDv4_t(i64 noundef %{{.*}})
// COER32:       define {{.*}}<4 x i32> @_Z18convert_ulong4_rteDv4_t(i64 noundef %{{.*}})
// FIXME: <4 x i16> should be coerced to i64 instead of double
// COER64:       define {{.*}}<4 x i64> @_Z18convert_ulong4_rteDv4_t(double noundef %{{.*}})
