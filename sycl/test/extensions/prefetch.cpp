// RUN: %clangxx -fsycl-device-only -S %s -o - | FileCheck %s

#include <sycl/sycl.hpp>

char data[] = {0, 1, 2, 3};

// CHECK: [[PREFETCH_STR:@.*]] = private unnamed_addr addrspace(1) constant [19 x i8] c"sycl-prefetch-hint\00", section "llvm.metadata"
// CHECK: [[PREFETCH_LVL0:@.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"0\00", section "llvm.metadata"
// CHECK: [[ANNOTATION1:@.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) [[PREFETCH_STR]], ptr addrspace(1) [[PREFETCH_LVL0]] }, section "llvm.metadata"
// CHECK: [[PREFETCH_LVL1:@.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"1\00", section "llvm.metadata"
// CHECK: [[ANNOTATION2:@.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) [[PREFETCH_STR]], ptr addrspace(1) [[PREFETCH_LVL1]] }, section "llvm.metadata"
// CHECK: [[PREFETCH_STR_NT:@.*]] = private unnamed_addr addrspace(1) constant [22 x i8] c"sycl-prefetch-hint-nt\00", section "llvm.metadata"
// CHECK: [[PREFETCH_LVL2:@.*]] = private unnamed_addr addrspace(1) constant [2 x i8] c"2\00", section "llvm.metadata"
// CHECK: [[ANNOTATION3:@.*]] = private unnamed_addr addrspace(1) constant { ptr addrspace(1), ptr addrspace(1) } { ptr addrspace(1) [[PREFETCH_STR_NT]], ptr addrspace(1) [[PREFETCH_LVL2]] }, section "llvm.metadata"

int main() {
  namespace syclex = sycl::ext::oneapi::experimental;
  sycl::queue q;
  void *dataPtr = &data;
  q.parallel_for(1, [=](sycl::id<1> idx) {
    // CHECK: [[CASTED:%.*]] = tail call spir_func noundef ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobal{{.*}}

    // CHECK: [[ANNOTATED1:%.*]] = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) [[CASTED]], ptr addrspace(1) {{.*}}, ptr addrspace(1) {{.*}}, i32 76, ptr addrspace(1) [[ANNOTATION1]])
    // CHECK: tail call spir_func void @_Z20__spirv_ocl_prefetch{{.*}}(ptr addrspace(1) noundef [[ANNOTATED1]], i64 noundef 1)
    syclex::prefetch(dataPtr);

    // CHECK: [[ANNOTATED2:%.*]] = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) [[CASTED]], ptr addrspace(1) {{.*}}, ptr addrspace(1) {{.*}}, i32 80, ptr addrspace(1) [[ANNOTATION2]])
    // CHECK: tail call spir_func void @_Z20__spirv_ocl_prefetch{{.*}}(ptr addrspace(1) noundef [[ANNOTATED2]], i64 noundef 1)
    syclex::prefetch(dataPtr, syclex::properties{syclex::prefetch_hint_L2});

    // CHECK: [[ANNOTATED3:%.*]] = tail call ptr addrspace(1) @llvm.ptr.annotation.p1.p1(ptr addrspace(1) [[CASTED]], ptr addrspace(1){{.*}}, ptr addrspace(1) {{.*}}, i32 80, ptr addrspace(1) [[ANNOTATION3]])
    // CHECK: tail call spir_func void @_Z20__spirv_ocl_prefetch{{.*}}(ptr addrspace(1) noundef [[ANNOTATED3]], i64 noundef 4)
    syclex::prefetch(dataPtr, 4,
                     syclex::properties{syclex::prefetch_hint_L3_nt});
  });
  q.wait();

  return 0;
}
