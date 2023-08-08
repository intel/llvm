// RUN: %clang_cc1 -triple spir64-unknown-unknown -disable-llvm-passes -fsycl-is-device -no-opaque-pointers -emit-llvm %s -o - | FileCheck %s

// This test checks that proper IR is generated for kernel field initialization, including
// 4 cases:
//   1. initialize pointer field with a global pointer
//   2. initialize a float field
//   3. initialize a BitInt field
//   4. initialize a field annotated with [[clang::annotate("...")]]
// It also checks that the kernel lambda is not inlined for FPGA, i.e. the kernel entry
// contains `call @NameOfCallOperator(...)`

// Note this is a temporary test for the FPGA-specific use model that will be
// replaced by kernel argument compile-time properties.

#include "Inputs/sycl.hpp"

// CHECK: @[[STR:.*]] = private unnamed_addr addrspace(1) constant [9 x i8] c"my_ann_1\00", section "llvm.metadata"


struct fooA {
    int *p;

// CHECK: define dso_local spir_kernel void @_ZTS4fooA(i32 addrspace(1)* {{.*}}%[[ARG:.*]])
// CHECK: %[[ARG_ADDR:.*]] = alloca i32 addrspace(1)*, align 8
// CHECK: %[[ARG_ADDR_CAST:.*]] = addrspacecast i32 addrspace(1)** %[[ARG_ADDR]] to i32 addrspace(1)* addrspace(4)*
// CHECK: store i32 addrspace(1)* %[[ARG]], i32 addrspace(1)* addrspace(4)* %[[ARG_ADDR_CAST]], align 8
// CHECK-DAG: %[[LOAD_ARG:.*]] = load i32 addrspace(1)*, i32 addrspace(1)* addrspace(4)* %[[ARG_ADDR_CAST]], align 8
// CHECK-DAG: %[[ADDR_CAST:.*]] = addrspacecast i32 addrspace(1)* %[[LOAD_ARG]] to i32 addrspace(4)*
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK: store i32 addrspace(4)* %[[ADDR_CAST]], i32 addrspace(4)* addrspace(4)* %[[GEP]], align 8
    fooA(int *_p) : p(_p) {}

// CHECK: call spir_func void @_ZNK4fooAclEv(
// CHECK-NEXT: ret void
    void operator()() const {}
};


struct fooB {
    float f;

// CHECK: define dso_local spir_kernel void @_ZTS4fooB({{.*}}%[[ARG:.*]])
// CHECK: %[[ARG_ADDR:.*]] = alloca float, align 4
// CHECK: %[[ARG_ADDR_CAST:.*]] = addrspacecast float* %[[ARG_ADDR]] to float addrspace(4)*
// CHECK: store float %[[ARG]], float addrspace(4)* %[[ARG_ADDR_CAST]], align 4
// CHECK-DAG: %[[LOAD_ARG:.*]] = load float, float addrspace(4)* %[[ARG_ADDR_CAST]], align 4
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK: store float %[[LOAD_ARG]], float addrspace(4)* %[[GEP]], align 4
    fooB(float _f) : f(_f) {}
    void operator()() const {}
};


struct bar {
  _BitInt(5) a;
};

struct fooC {
    bar b;

// CHECK: define dso_local spir_kernel void @_ZTS4fooC(%struct.bar* {{.*}}%[[ARG:.*]])
// CHECK: %[[ARG_AS_CAST:.*]] = addrspacecast %struct.bar* %[[ARG]] to %struct.bar addrspace(4)*
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK-DAG: %[[GEP_CAST:.*]] = bitcast %struct.bar addrspace(4)* %[[GEP]] to i8 addrspace(4)*
// CHECK-DAG: %[[ARG_CAST:.*]] = bitcast %struct.bar addrspace(4)* %[[ARG_AS_CAST]] to i8 addrspace(4)*
// CHECK: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 1 %[[GEP_CAST]], i8 addrspace(4)* align 1 %[[ARG_CAST]], i64 1, i1 false)
    fooC(bar _b) : b(_b) {}
    void operator()() const {}
};


struct fooD {
    [[clang::annotate("my_ann_1")]]
    int n;

// CHECK: define dso_local spir_kernel void @_ZTS4fooD(i32 {{.*}}%[[ARG:.*]])
// CHECK: %[[ARG_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG_ADDR_AS_CAST:.*]] = addrspacecast i32* %[[ARG_ADDR]] to i32 addrspace(4)*
// CHECK: store i32 %[[ARG]], i32 addrspace(4)* %[[ARG_ADDR_AS_CAST]], align 4
// CHECK-DAG: %[[LOAD_ARG:.*]] = load i32, i32 addrspace(4)* %[[ARG_ADDR_AS_CAST]], align 4
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK-DAG: %[[ANNOTATED_PTR:.*]] = call i32 addrspace(4)* @llvm.ptr.annotation.p4i32.p1i8(i32 addrspace(4)* %[[GEP]], {{.*}} @[[STR]],
// CHECK: store i32 %[[LOAD_ARG]], i32 addrspace(4)* %[[ANNOTATED_PTR]], align 4
    fooD(int _n) : n(_n) {}

    void operator()() const {}
};

int main() {
  sycl::handler h;
  h.single_task(fooA{nullptr});
  h.single_task(fooB{2.0});
  h.single_task(fooC{{3}});
  h.single_task(fooD{3});
  return 0;
}
