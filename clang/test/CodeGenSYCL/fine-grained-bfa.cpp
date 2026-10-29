// RUN: %clang -cc1 -fsycl-is-device %s -triple spir64 -aux-triple x86_64-windows -ffine-grained-bitfield-accesses -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-FGBA
// RUN: %clang -cc1 -fsycl-is-device %s -triple spir64 -aux-triple x86_64-windows -fno-fine-grained-bitfield-accesses -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-NOFGBA
// RUN: %clang -cc1 -fsycl-is-device %s -triple spir64 -aux-triple x86_64-windows -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-NOFGBA
//
// RUN: %clang -cc1 -fsycl-is-device %s -triple spir64 -aux-triple x86_64-linux -ffine-grained-bitfield-accesses -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-FGBA-LX
// RUN: %clang -cc1 -fsycl-is-device %s -triple spir64 -aux-triple x86_64-linux -fno-fine-grained-bitfield-accesses -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-NOFGBA-LX
// RUN: %clang -cc1 -fsycl-is-device %s -triple spir64 -aux-triple x86_64-linux -emit-llvm -o - | FileCheck %s -check-prefixes=CHECK-NOFGBA-LX

// This test checks if the option -ffine-grained-bitfield-accesses takes effect.

enum GType {
  GTY_INSTANCE_ARRAY = 24,
};
enum GTypeMask : unsigned long {
  MTY_INSTANCE_ARRAY     = 1ul << GTY_INSTANCE_ARRAY,
};
struct Device;

class __attribute__((aligned(16))) Geometry {
public:
  Device *device;              // 16
  struct {                     // 60
    GType gtype : 8;
    unsigned state : 2;
    bool enabled : 1;
    bool argumentFilterEnabled : 1;
  };
  // real getTypeMask(): note it is a SHIFT, not an equality compare
  __attribute__((always_inline)) GTypeMask getTypeMask() const {
    return (GTypeMask)(1 << gtype);
  }
};

// CHECK-FGBA:    %struct.anon = type { i8, i8, [2 x i8], i8, [3 x i8] }
// CHECK-NOFGBA:  %struct.anon = type { i32, i8 }
// CHECK-FGBA-LX: %struct.anon = type { i8, i8, [2 x i8] }
// CHECK-NOFGBA-LX: %struct.anon = type { i16, [2 x i8] }

SYCL_EXTERNAL void *func1(Geometry *g);
SYCL_EXTERNAL void *func2(Geometry *g);
SYCL_EXTERNAL bool trace(void *object);

__attribute__((always_inline))
static bool instance(Geometry *geom, bool flag) {
  void *object = flag ? func1(geom) : func2(geom);
  return trace(object);
}

SYCL_EXTERNAL bool dispatch(Geometry *geom, unsigned feature_mask) {
  return instance(geom, geom->getTypeMask() & MTY_INSTANCE_ARRAY);
// CHECK-FGBA: %bf.load.i = load i8, ptr addrspace(4) %2, align 8
// CHECK-FGBA: %bf.cast.i = zext i8 %bf.load.i to i32
// CHECK-FGBA: %shl.i = shl i32 1, %bf.cast.i
// CHECK-FGBA-LX: %bf.load.i = load i8, ptr addrspace(4) %2, align 8
// CHECK-FGBA-LX: %bf.cast.i = zext i8 %bf.load.i to i32
// CHECK-FGBA-LX: %shl.i = shl i32 1, %bf.cast.i
// CHECK-NOFGBA: %bf.load.i = load i32, ptr addrspace(4) %2, align 8
// CHECK-NOFGBA: %bf.clear.i = and i32 %bf.load.i, 255
// CHECK-NOFGBA: %shl.i = shl i32 1, %bf.clear.i
// CHECK-NOFGBA-LX: %bf.load.i = load i16, ptr addrspace(4) %2, align 8
// CHECK-NOFGBA-LX: %bf.clear.i = and i16 %bf.load.i, 255
// CHECK-NOFGBA-LX: %bf.cast.i = zext i16 %bf.clear.i to i32
// CHECK-NOFGBA-LX: %shl.i = shl i32 1, %bf.cast.i
}
