// RUN: clang++ -fsycl -fsycl-device-only -O0 -w -S -emit-llvm -o - %s | FileCheck %s --check-prefixes CHECK,CHECK-O0
// RUN: clang++ -fsycl -fsycl-device-only -O1 -w -S -emit-llvm -o - %s | FileCheck %s --check-prefixes CHECK,CHECK-O1
// RUN: clang++ -fsycl -fsycl-device-only -O2 -w -S -emit-llvm -o - %s | FileCheck %s --check-prefixes CHECK,CHECK-O2
// RUN: clang++ -fsycl -fsycl-device-only -O3 -w -S -emit-llvm -o - %s | FileCheck %s --check-prefixes CHECK,CHECK-O3

// CHECK:    define dso_local spir_func void @_Z4funcv()
// CHECK-SAME:                                           #[[ATTRS:.*]]
SYCL_EXTERNAL void func() {
  return;
}

// CHECK-O0:    attributes #[[ATTRS:.*]] = { {{.*}} "sycl-optlevel"="0"
// CHECK-O1:    attributes #[[ATTRS:.*]] = { {{.*}} "sycl-optlevel"="1"
// CHECK-O2:    attributes #[[ATTRS:.*]] = { {{.*}} "sycl-optlevel"="2"
// CHECK-O3:    attributes #[[ATTRS:.*]] = { {{.*}} "sycl-optlevel"="3"
