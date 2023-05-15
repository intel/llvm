// RUN: %clangxx %s -fsycl-device-only -fintelfpga -S -o %t.ll
// RUN: FileCheck %s --input-file %t.ll

// This test checks that proper IR is generated for kernel field initialization, including
// 3 cases:
//   1. initialize pointer field with a global pointer
//   2. initialize a float field
//   3. initialize a BitInt field
// It also checks that the kernel lambda is not inlined for FPGA, i.e. the kernel entry
// contains `call @NameOfCallOperator(...)`

// Note this is a tempoary test for the FPGA-specific use model that will be
// replaced by kernel argument compile-time properties.

#include "Inputs/sycl.hpp"


struct fooA {
    int *p;

// CHECK: define dso_local spir_kernel void @_ZTS4fooA(i32 addrspace(1)* {{.*}}%[[ARG:.*]])
// CHECK: %[[ARG_ADDR:.*]] = alloca i32 addrspace(1)*, align 8
// CHECK: store i32 addrspace(1)* %[[ARG]], i32 addrspace(1)** %[[ARG_ADDR]], align 8
// CHECK-DAG: %[[LOAD_ARG:.*]] = load i32 addrspace(1)*, i32 addrspace(1)** %[[ARG_ADDR]], align 8
// CHECK-DAG: %[[ADDR_CAST:.*]] = addrspacecast i32 addrspace(1)* %[[LOAD_ARG]] to i32 addrspace(4)*
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK: store i32 addrspace(4)* %[[ADDR_CAST]], i32 addrspace(4)** %[[GEP]], align 8
    fooA(int *_p) : p(_p) {}


// CHECK: call spir_func void @_ZNK4fooAclEv(
// CHECK: call void @__itt_offload_wi_finish_wrapper()
    void operator()() const {}
};


struct fooB {
    float f;

// CHECK: define dso_local spir_kernel void @_ZTS4fooB({{.*}}%[[ARG:.*]])
// CHECK: %[[ARG_ADDR:.*]] = alloca float, align 4
// CHECK: store float %[[ARG]], float* %[[ARG_ADDR]], align 4
// CHECK-DAG: %[[LOAD_ARG:.*]] = load float, float* %[[ARG_ADDR]], align 4
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK: store float %[[LOAD_ARG]], float* %[[GEP]], align 4
    fooB(float _f) : f(_f) {}
    void operator()() const {}
};


struct bar {
  _BitInt(5) a;
};

struct fooC {
    bar b;

// CHECK: define dso_local spir_kernel void @_ZTS4fooC(%struct.bar* {{.*}}%[[ARG:.*]])
// CHECK-DAG: %[[GEP:.*]] = getelementptr inbounds
// CHECK-DAG: %[[GEP_CAST:.*]] = bitcast %struct.bar* %[[GEP]] to i8*
// CHECK-DAG: %[[ARG_CAST:.*]] = bitcast %struct.bar* %[[ARG]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %[[GEP_CAST]], i8* align 1 %[[ARG_CAST]], i64 1, i1 false)
    fooC(bar _b) : b(_b) {}
    void operator()() const {}
};

int main() {
  sycl::handler h;
  h.single_task(fooA{nullptr});
  h.single_task(fooB{2.0});
  h.single_task(fooC{{3}});
  return 0;
}
