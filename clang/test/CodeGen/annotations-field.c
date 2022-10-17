// RUN: %clang_cc1 -emit-llvm -o - %s | FileCheck %s
// END.

// CHECK: private unnamed_addr constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr constant [8 x i8] c"v_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr constant [8 x i8] c"w_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr constant [8 x i8] c"w_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr constant [8 x i8] c"f_ann_{{.}}\00", section "llvm.metadata"
// CHECK: private unnamed_addr constant [8 x i8] c"f_ann_{{.}}\00", section "llvm.metadata"

struct foo {
    int v __attribute__((annotate("v_ann_0"))) __attribute__((annotate("v_ann_1")));
    char w __attribute__((annotate("w_ann_0"))) __attribute__((annotate("w_ann_1")));
    float f __attribute__((annotate("f_ann_0"))) __attribute__((annotate("f_ann_1")));
};

static struct foo gf;

int main(int argc, char **argv) {
    struct foo f;
    f.v = argc;
// CHECK: getelementptr inbounds %struct.foo, ptr %f, i32 0, i32 0
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 12, ptr null)
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 12, ptr null)
    f.w = 42;
// CHECK: getelementptr inbounds %struct.foo, ptr %f, i32 0, i32 1
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 13, ptr null)
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 13, ptr null)
    f.f = 0;
// CHECK: getelementptr inbounds %struct.foo, ptr %f, i32 0, i32 2
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 14, ptr null)
// CHECK-NEXT: call ptr @llvm.ptr.annotation.p0({{.*}}str{{.*}}str{{.*}}i32 14, ptr null)
    gf.v = argc;
// CHECK: call ptr @llvm.ptr.annotation.p0(ptr @gf, {{.*}}str{{.*}}str{{.*}}i32 12, ptr null)
    gf.w = 42;
// CHECK: call ptr @llvm.ptr.annotation.p0(ptr getelementptr inbounds (%struct.foo, ptr @gf, i32 0, i32 1), {{.*}}str{{.*}}str{{.*}}i32 13, ptr null)
    gf.f = 0;
// CHECK: call ptr @llvm.ptr.annotation.p0(ptr getelementptr inbounds (%struct.foo, ptr @gf, i32 0, i32 2), {{.*}}str{{.*}}str{{.*}}i32 14, ptr null)
    return 0;
}
