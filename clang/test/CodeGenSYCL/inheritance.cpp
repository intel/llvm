// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

class second_base {
public:
  int *e;
  int *arr[2];
  second_base(int *E) : e(E) {}
};

class InnerFieldBase {
public:
  int d;
};
class InnerField : public InnerFieldBase {
  int c;
};

struct base {
public:
  int b;
  InnerField obj;
};

struct derived : base, second_base {
  int a;
  derived() : second_base(nullptr) {}
  void operator()() const {
  }
};

int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &cgh) {
    derived f{};
    cgh.single_task(f);
  });

  return 0;
}

// CHECK: %struct.base = type { i32, %class.InnerField }
// CHECK: %class.InnerField = type { %class.InnerFieldBase, i32 }
// CHECK: %class.InnerFieldBase = type { i32 }
// CHECK: %class.__generated_second_base = type { ptr addrspace(1), [2 x ptr addrspace(1)] }
// CHECK: %struct.derived = type <{ %struct.base, [4 x i8], %class.second_base, i32, [4 x i8] }>
// CHECK: %class.second_base = type { ptr addrspace(4), [2 x ptr addrspace(4)] }

// Check kernel paramters
// CHECK: define {{.*}}spir_kernel void @{{.*}}derived
// CHECK-SAME: ptr noundef byval(%struct.base) align 4 %_arg__base
// CHECK-SAME: ptr noundef byval(%class.__generated_second_base) align 8 %_arg__base1
// CHECK-SAME: i32 noundef %_arg_a

// Check allocas for kernel parameters and local functor object
// CHECK: %[[ARG_A_ALLOCA:[a-zA-Z0-9_.]+]] = alloca i32, align 4
// CHECK: %[[LOCAL_OBJECT_ALLOCA:[a-zA-Z0-9_.]+]] = alloca %struct.derived, align 8
// CHECK: %[[ARG_A:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[ARG_A_ALLOCA]] to ptr addrspace(4)
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[LOCAL_OBJECT_ALLOCA]] to ptr addrspace(4)
// CHECK: %[[ARG_BASE:[a-zA-Z0-9_.]+]] = addrspacecast ptr %_arg__base to ptr addrspace(4)
// CHECK: %[[ARG_BASE1:[a-zA-Z0-9_.]+]] = addrspacecast ptr %_arg__base1 to ptr addrspace(4)
// CHECK: store i32 %_arg_a, ptr addrspace(4) %[[ARG_A]], align 4

// Initialize 'base' subobject
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %[[LOCAL_OBJECT]], ptr addrspace(4) align 4 %[[ARG_BASE]], i64 12, i1 false)

// Initialize 'second_base' subobject
// First, derived-to-base cast with offset:
// CHECK: %[[OFFSET_CALC:.*]] = getelementptr inbounds i8, ptr addrspace(4) %[[LOCAL_OBJECT]], i64 16
// Initialize 'second_base'
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %[[OFFSET_CALC]], ptr addrspace(4) align 8 %[[ARG_BASE1]], i64 24, i1 false)

// Initialize field 'a'
// CHECK: %[[GEP_A:[a-zA-Z0-9]+]] = getelementptr inbounds nuw %struct.derived, ptr addrspace(4) %[[LOCAL_OBJECT]], i32 0, i32 3
// CHECK: %[[LOAD_A:[0-9]+]] = load i32, ptr addrspace(4) %[[ARG_A]], align 4
// CHECK: store i32 %[[LOAD_A]], ptr addrspace(4) %[[GEP_A]]

