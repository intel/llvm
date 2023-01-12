// RUN: %clangxx -fsycl -fsycl-targets=spir64 %s -S -emit-llvm -o- | FileCheck -check-prefix=CHECK-FINE %s
// RUN: %clangxx -fsycl -fsycl-targets=spir64 -fsycl-disable-fine-grained-bitfield-accesses %s -S -emit-llvm -o- | FileCheck -check-prefix=CHECK-COARSE %s
// RUN: %clangxx -fsycl %s -S -emit-llvm -o- | FileCheck -check-prefix=CHECK-FINE %s
// RUN: %clangxx %s -S -emit-llvm -o- | FileCheck -check-prefix=CHECK-COARSE %s

// CHECK-FINE: %struct.with_bitfield = type { i32, i32, i32, i32 }
// CHECK-COARSE: %struct.with_bitfield = type { i128 }
//
// Tests if fine grained access for SPIR targets is working

struct with_bitfield {
    unsigned int a : 32;
    unsigned int b : 32;
    unsigned int c : 32;
    unsigned int d : 32;
};

int main() {
    with_bitfield A;
    return 0;
}

