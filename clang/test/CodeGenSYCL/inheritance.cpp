// RUN:  %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o - | FileCheck %s

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
  cl::sycl::queue q;

  q.submit([&](cl::sycl::handler &cgh) {
    derived f{};
    cgh.single_task(f);
  });

  return 0;
}

// Check kernel paramters
// CHECK: define {{.*}}spir_kernel void @{{.*}}derived(ptr noundef byval(%struct.base) align 4 %_arg__base, ptr noundef byval(%struct.__wrapper_class) align 8 %_arg_e, i32 noundef %_arg_a)

// Check alloca for kernel paramters
// CHECK: %[[ARG_AA:[a-zA-Z0-9_.]+]] = alloca i32, align 4
// Check alloca for local functor object
// CHECK: %[[LOCAL_OBJECT:[a-zA-Z0-9_.]+]] = alloca %struct.derived, align 8
// CHECK: %[[ARG_A:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[ARG_AA]] to ptr addrspace(4)
// CHECK: %[[BASE_TO_PTR:[a-zA-Z0-9_.]+]] = addrspacecast ptr %[[LOCAL_OBJECT]] to ptr addrspace(4)
// CHECK: store i32 %_arg_a, ptr addrspace(4) %[[ARG_A]], align 4

// Initialize 'base' subobject
// CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 8 %[[BASE_TO_PTR]], ptr addrspace(4) align 4 %_arg__base.ascast, i64 12, i1 false)

// Initialize 'second_base' subobject
// First, derived-to-base cast with offset:
// CHECK: %[[OFFSET_CALC:.*]] = getelementptr inbounds i8, ptr addrspace(4) %[[LOCAL_OBJECT]].ascast, i64 16
// Initialize 'second_base::e'
// CHECK: %[[SECOND_BASE_PTR:.*]] = getelementptr inbounds %class.second_base, ptr addrspace(4) %[[OFFSET_CALC]], i32 0, i32 0
// CHECK: %[[PTR_TO_WRAPPER:.*]] = getelementptr inbounds %struct.__wrapper_class, ptr addrspace(4) %_arg_e.ascast, i32 0, i32 0
// CHECK: %[[LOAD_PTR:.*]] = load ptr addrspace(1), ptr addrspace(4) %[[PTR_TO_WRAPPER]]
// CHECK: %[[AS_CAST:.*]] = addrspacecast ptr addrspace(1) %[[LOAD_PTR]] to ptr addrspace(4)
// CHECK: store ptr addrspace(4) %[[AS_CAST]], ptr addrspace(4) %[[SECOND_BASE_PTR]]

// Initialize field 'a'
// CHECK: %[[GEP_A:[a-zA-Z0-9]+]] = getelementptr inbounds %struct.derived, ptr addrspace(4) %[[LOCAL_OBJECT]].ascast, i32 0, i32 3
// CHECK: %[[LOAD_A:[0-9]+]] = load i32, ptr addrspace(4) %[[ARG_A]], align 4
// CHECK: store i32 %[[LOAD_A]], ptr addrspace(4) %[[GEP_A]]
