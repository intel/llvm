// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

#include "Inputs/sycl.hpp"

class second_base {
public:
  int *e;
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
// CHECK: %class.__generated_second_base = type { i32 addrspace(1)* }
// CHECK: %struct.derived = type <{ %struct.base, [4 x i8], %class.second_base, i32, [4 x i8] }>
// CHECK: %class.second_base = type { i32 addrspace(4)* }

// Check kernel paramters
// CHECK: define {{.*}}spir_kernel void @{{.*}}derived
// CHECK-SAME: %struct.base* noundef byval(%struct.base) align 4 %_arg__base
// CHECK-SAME: %class.__generated_second_base* noundef byval(%class.__generated_second_base) align 8 %_arg__base1
// CHECK-SAME: i32 noundef %_arg_a

// Check allocas for kernel parameters and local functor object
// CHECK: %[[ARG_A_ALLOCA:[a-zA-Z0-9_.]+]] = alloca i32, align 4
// CHECK: %[[LOCAL_OBJECT_ALLOCA:[a-zA-Z0-9_.]+]] = alloca %struct.derived, align 8
// CHECK: %[[ARG_A:[a-zA-Z0-9_.]+]] = addrspacecast i32* %[[ARG_A_ALLOCA]] to i32 addrspace(4)*
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = addrspacecast %struct.derived* %[[LOCAL_OBJECT_ALLOCA]] to %struct.derived addrspace(4)*
// CHECK: %[[ARG_BASE:[a-zA-Z0-9_.]+]] = addrspacecast %struct.base* %_arg__base to %struct.base addrspace(4)*
// CHECK: %[[ARG_BASE1:[a-zA-Z0-9_.]+]] = addrspacecast %class.__generated_second_base* %_arg__base1 to %class.__generated_second_base addrspace(4)*
// CHECK: store i32 %_arg_a, i32 addrspace(4)* %[[ARG_A]], align 4

// Initialize 'base' subobject
// CHECK: %[[DERIVED_TO_BASE:.*]] = bitcast %struct.derived addrspace(4)* %[[LOCAL_OBJECT]] to %struct.base addrspace(4)*
// CHECK: %[[BASE_TO_PTR:.*]] = bitcast %struct.base addrspace(4)* %[[DERIVED_TO_BASE]] to i8 addrspace(4)*
// CHECK: %[[PARAM_TO_PTR:.*]] = bitcast %struct.base addrspace(4)* %[[ARG_BASE]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %[[BASE_TO_PTR]], i8 addrspace(4)* align 4 %[[PARAM_TO_PTR]], i64 12, i1 false)

// Initialize 'second_base' subobject
// First, derived-to-base cast with offset:
// CHECK: %[[DERIVED_PTR:.*]] = bitcast %struct.derived addrspace(4)* %[[LOCAL_OBJECT]] to i8 addrspace(4)*
// CHECK: %[[OFFSET_CALC:.*]] = getelementptr inbounds i8, i8 addrspace(4)* %[[DERIVED_PTR]], i64 16
// CHECK: %[[TO_SECOND_BASE:.*]] = bitcast i8 addrspace(4)* %[[OFFSET_CALC]] to %class.second_base addrspace(4)*
// CHECK: %[[GEN_TO_SECOND_BASE:.*]] = bitcast %class.__generated_second_base addrspace(4)* %[[ARG_BASE1]] to %class.second_base addrspace(4)*
// CHECK: %[[TO:.*]] = bitcast %class.second_base addrspace(4)* %[[TO_SECOND_BASE]] to i8 addrspace(4)*
// CHECK: %[[FROM:.*]] = bitcast %class.second_base addrspace(4)* %[[GEN_TO_SECOND_BASE]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %[[TO]], i8 addrspace(4)* align 8 %[[FROM]], i64 8, i1 false)


// Initialize field 'a'
// CHECK: %[[GEP_A:[a-zA-Z0-9]+]] = getelementptr inbounds %struct.derived, %struct.derived addrspace(4)* %[[LOCAL_OBJECT]], i32 0, i32 3
// CHECK: %[[LOAD_A:[0-9]+]] = load i32, i32 addrspace(4)* %[[ARG_A]], align 4
// CHECK: store i32 %[[LOAD_A]], i32 addrspace(4)* %[[GEP_A]]
