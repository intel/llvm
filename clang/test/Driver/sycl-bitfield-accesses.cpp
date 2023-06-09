// RUN: %clangxx -fsycl-device-only %s -S -emit-llvm -o- | FileCheck %s

// CHECK: %struct.with_bitfield = type { i32, i32, i32, i32 }
//
// Tests if fine grained access for SPIR targets is enabled.

struct with_bitfield {
    unsigned int a : 32;
    unsigned int b : 32;
    unsigned int c : 32;
    unsigned int d : 32;
};

SYCL_EXTERNAL unsigned int foo(with_bitfield A) {
  return A.a + A.b + A.c + A.d;
}

struct with_bitfield_small_ints {
    unsigned int a : 5;
    unsigned int b : 7;
    unsigned int c : 20;
    unsigned int d : 32;
};

SYCL_EXTERNAL unsigned int boo(with_bitfield_small_ints A) {
  return A.a + A.b + A.c + A.d;
}
