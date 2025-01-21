// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -emit-llvm -o - %s | FileCheck %s

// This test checks that clang emits @llvm.ptr.annotation intrinsic correctly
// when attribute annotate is applied to a struct field of integer type.

// CHECK: private unnamed_addr addrspace(1) constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr addrspace(1) constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr addrspace(1) constant [8 x i8] c"w_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr addrspace(1) constant [8 x i8] c"w_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr addrspace(1) constant [8 x i8] c"f_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr addrspace(1) constant [8 x i8] c"f_ann_{{.}}\00", section "llvm.metadata"

struct foo {
    int v __attribute__((annotate("v_ann_0"))) __attribute__((annotate("v_ann_1")));
    char w __attribute__((annotate("w_ann_0"))) __attribute__((annotate("w_ann_1")));
    float f __attribute__((annotate("f_ann_0"))) __attribute__((annotate("f_ann_1")));
};

int __attribute__((sycl_device)) foo() {
    struct foo f;
    f.v = 1;
// CHECK: getelementptr inbounds nuw %struct.foo, ptr addrspace(4) %{{.*}}, i32 0, i32 0
// CHECK-NEXT: call ptr addrspace(4) @llvm.ptr.annotation.p4.p1({{.*}}str{{.*}}str{{.*}}i32 14, ptr addrspace(1) null)
// CHECK-NEXT: call ptr addrspace(4) @llvm.ptr.annotation.p4.p1({{.*}}str{{.*}}str{{.*}}i32 14, ptr addrspace(1) null)
    f.w = 42;
// CHECK: getelementptr inbounds nuw %struct.foo, ptr addrspace(4) %{{.*}}, i32 0, i32 1
// CHECK-NEXT: call ptr addrspace(4) @llvm.ptr.annotation.p4.p1({{.*}}str{{.*}}str{{.*}}i32 15, ptr addrspace(1) null)
// CHECK-NEXT: call ptr addrspace(4) @llvm.ptr.annotation.p4.p1({{.*}}str{{.*}}str{{.*}}i32 15, ptr addrspace(1) null)
    f.f = 0;
// CHECK: getelementptr inbounds nuw %struct.foo, ptr addrspace(4) %{{.*}}, i32 0, i32 2
// CHECK-NEXT: call ptr addrspace(4) @llvm.ptr.annotation.p4.p1({{.*}}str{{.*}}str{{.*}}i32 16, ptr addrspace(1) null)
// CHECK-NEXT: call ptr addrspace(4) @llvm.ptr.annotation.p4.p1({{.*}}str{{.*}}str{{.*}}i32 16, ptr addrspace(1) null)
    return 0;
}
