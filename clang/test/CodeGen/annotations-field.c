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
// CHECK: getelementptr inbounds %struct.foo, %struct.foo* %f, i32 0, i32 0
// CHECK-NEXT: call i32* @llvm.ptr.annotation.p0i32({{.*}}str{{.*}}str{{.*}}i32 12, i8* null)
// CHECK-NEXT: call i32* @llvm.ptr.annotation.p0i32({{.*}}str{{.*}}str{{.*}}i32 12, i8* null)
    f.w = 42;
// CHECK: getelementptr inbounds %struct.foo, %struct.foo* %f, i32 0, i32 1
// CHECK-NEXT: call i8* @llvm.ptr.annotation.p0i8({{.*}}str{{.*}}str{{.*}}i32 13, i8* null)
// CHECK-NEXT: call i8* @llvm.ptr.annotation.p0i8({{.*}}str{{.*}}str{{.*}}i32 13, i8* null)
    f.f = 0;
// CHECK: getelementptr inbounds %struct.foo, %struct.foo* %f, i32 0, i32 2
// CHECK-NEXT: bitcast float* {{.*}} to i8*
// CHECK-NEXT: call i8* @llvm.ptr.annotation.p0i8({{.*}}str{{.*}}str{{.*}}i32 14, i8* null)
// CHECK-NEXT: bitcast i8* {{.*}} to float*
// CHECK-NEXT: bitcast float* {{.*}} to i8*
// CHECK-NEXT: call i8* @llvm.ptr.annotation.p0i8({{.*}}str{{.*}}str{{.*}}i32 14, i8* null)
// CHECK-NEXT: bitcast i8* {{.*}} to float*
    gf.v = argc;
// CHECK: call i32* @llvm.ptr.annotation.p0i32(i32* getelementptr inbounds (%struct.foo, %struct.foo* @gf, i32 0, i32 0), {{.*}}str{{.*}}str{{.*}}i32 12, i8* null)
    gf.w = 42;
// CHECK: call i8* @llvm.ptr.annotation.p0i8(i8* getelementptr inbounds (%struct.foo, %struct.foo* @gf, i32 0, i32 1), {{.*}}str{{.*}}str{{.*}}i32 13, i8* null)
    gf.f = 0;
// CHECK: call i8* @llvm.ptr.annotation.p0i8(i8* bitcast (float* getelementptr inbounds (%struct.foo, %struct.foo* @gf, i32 0, i32 2) to i8*), {{.*}}str{{.*}}str{{.*}}i32 14, i8* null)
    return 0;
}
