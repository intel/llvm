// RUN: %clang_cc1 -triple spir64-unknown-linux-sycldevice -std=c++11 -fsycl-is-device -disable-llvm-passes -S -emit-llvm -x c++ %s -o - | FileCheck %s

struct st {
  int a;
  float b;
};
// CHECK: %[[T_ST:struct[a-zA-Z0-9_.]*.st]] = type { i32, float }

union un {
  int a;
  char c[4];
};
// CHECK: %[[T_UN:union[a-zA-Z0-9_.]*.un]] = type { i32 }

class A {
public:
  A(int a) {
    m_val = a;
  }
  A(const A &a) {
    m_val = a.m_val;
  }
private:
    int m_val;
};
// CHECK: %[[T_CL:class[a-zA-Z0-9_.]*.A]] = type { i32 }

typedef int myInt;

// CHECK: @.str = private unnamed_addr constant [25 x i8] c"__builtin_intel_fpga_reg\00", section "llvm.metadata"

void foo() {
  int a=123;
  myInt myA = 321;
  int b = __builtin_intel_fpga_reg(a);
// CHECK: %[[V_A1:[0-9]+]] = load i32, i32* %a, align 4, !tbaa !9
// CHECK-NEXT: %[[V_A2:[0-9]+]] = call i32 @llvm.annotation.i32(i32 %[[V_A1]], [[BIFR_STR:i8\* getelementptr inbounds \(\[25 x i8\], \[25 x i8\]\* @.str, i32 0, i32 0\),]]
// CHECK-NEXT: store i32 %[[V_A2]], i32* %b, align 4, !tbaa !9
  int myB = __builtin_intel_fpga_reg(myA);
// CHECK: %[[V_MYA1:[0-9]+]] = load i32, i32* %myA
// CHECK-NEXT: %[[V_MYA2:[0-9]+]] = call i32 @llvm.annotation.i32(i32 %[[V_MYA1]], [[BIFR_STR]]
// CHECK-NEXT: store i32 %[[V_MYA2]], i32* %myB, align 4, !tbaa !9
  int c = __builtin_intel_fpga_reg(2.0f);
// CHECK: %[[V_CF1:[0-9]+]] = call i32 @llvm.annotation.i32(i32 1073741824, [[BIFR_STR]]
// CHECK-NEXT: %[[V_FBITCAST:[0-9]+]] = bitcast i32 %[[V_CF1]] to float
// CHECK-NEXT: %[[V_CF2:conv]] = fptosi float %[[V_FBITCAST]] to i32
// CHECK-NEXT: store i32 %[[V_CF2]], i32* %c, align 4, !tbaa !9
  int d = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( b+12 ));
// CHECK: %[[V_B1:[0-9]+]] = load i32, i32* %b
// CHECK-NEXT: %[[V_B2:add]] = add nsw i32 %[[V_B1]], 12
// CHECK-NEXT: %[[V_B3:[0-9]+]] = call i32 @llvm.annotation.i32(i32 %[[V_B2]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_B4:[0-9]+]] = call i32 @llvm.annotation.i32(i32 %[[V_B3]], [[BIFR_STR]]
// CHECK-NEXT: store i32 %[[V_B4]], i32* %d, align 4, !tbaa !9
  int e = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( a+b ));
// CHECK: %[[V_AB1:[0-9]+]] = load i32, i32* %a
// CHECK-NEXT: %[[V_AB2:[0-9]+]] = load i32, i32* %b
// CHECK-NEXT: %[[V_AB3:add[0-9]+]] = add nsw i32 %[[V_AB1]], %[[V_AB2]]
// CHECK-NEXT: %[[V_AB4:[0-9]+]] = call i32 @llvm.annotation.i32(i32 %[[V_AB3]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_AB5:[0-9]+]] = call i32 @llvm.annotation.i32(i32 %[[V_AB4]], [[BIFR_STR]]
// CHECK-NEXT: store i32 %[[V_AB5]], i32* %e, align 4, !tbaa !9
  int f;
  f = __builtin_intel_fpga_reg(a);
// CHECK: %[[V_F1:[0-9]+]] = load i32, i32* %a
// CHECK-NEXT: %[[V_F2:[0-9]+]] = call i32 @llvm.annotation.i32(i32 %[[V_F1]], [[BIFR_STR]]
// CHECK-NEXT: store i32 %[[V_F2]], i32* %f, align 4, !tbaa !9

  struct st i = {1, 5.0f};
  struct st i2 = i;
  struct st ii = __builtin_intel_fpga_reg(i);
// CHECK: %[[V_TI1:[0-9]+]] = bitcast %[[T_ST]]* %agg-temp to i8*
// CHECK-NEXT: %[[V_I:[0-9]+]] = bitcast %[[T_ST]]* %i to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TI1]], i8* align 4 %[[V_I]], i64 8, i1 false), !tbaa.struct !11
// CHECK-NEXT: %[[V_TI2:[0-9]+]] = bitcast %[[T_ST]]* %agg-temp to i8*
// CHECK-NEXT: %[[V_TI3:[0-9]+]] = call i8* @llvm.ptr.annotation.p0i8(i8* %[[V_TI2]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_TI4:[0-9]+]] = bitcast i8* %[[V_TI3]] to %[[T_ST]]*
// CHECK-NEXT: %[[V_II:[0-9]+]] = bitcast %[[T_ST]]* %ii to i8*
// CHECK-NEXT: %[[V_TI5:[0-9]+]] = bitcast %[[T_ST]]* %[[V_TI4]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_II]], i8* align 4 %[[V_TI5]], i64 8, i1 false)
  struct st iii;
  iii = __builtin_intel_fpga_reg(ii);
// CHECK: %[[V_TII1:[0-9]+]] = bitcast %[[T_ST]]* %agg-temp2 to i8*
// CHECK-NEXT: %[[V_II:[0-9]+]] = bitcast %[[T_ST]]* %ii to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TII1]], i8* align 4 %[[V_II]], i64 8, i1 false), !tbaa.struct !11
// CHECK-NEXT: %[[V_TII2:[0-9]+]] = bitcast %[[T_ST]]* %agg-temp2 to i8*
// CHECK-NEXT: %[[V_TII3:[0-9]+]] = call i8* @llvm.ptr.annotation.p0i8(i8* %[[V_TII2]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_TII4:[0-9]+]] = bitcast i8* %[[V_TII3]] to %[[T_ST]]*
// CHECK-NEXT: %[[V_TII5:[0-9]+]] = bitcast %[[T_ST]]* %ref.tmp to i8*
// CHECK-NEXT: %[[V_TII6:[0-9]+]] = bitcast %[[T_ST]]* %[[V_TII4]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TII5]], i8* align 4 %[[V_TII6]], i64 8, i1 false)
// CHECK-NEXT: %[[V_TIII:[0-9]+]] = bitcast %[[T_ST]]* %iii to i8*
// CHECK-NEXT: %[[V_TII7:[0-9]+]] = bitcast %[[T_ST]]* %ref.tmp to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TIII]], i8* align 4 %[[V_TII7]], i64 8, i1 false), !tbaa.struct !11

  struct st *iiii = __builtin_intel_fpga_reg(&iii);
// CHECK: %[[V_T3I0:[0-9]+]] = ptrtoint %[[T_ST]]* %iii to i64
// CHECK-NEXT: %[[V_T3I1:[0-9]+]] = call i64 @llvm.annotation.i64(i64 %[[V_T3I0]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_T3I2:[0-9]+]] = inttoptr i64 %[[V_T3I1]] to %[[T_ST]]*
// CHECK-NEXT: %[[V_T3I3:[0-9]+]] = addrspacecast %[[T_ST]]* %[[V_T3I2]] to %[[T_ST]] addrspace(4)*
// CHECK-NEXT: store %[[T_ST]] addrspace(4)* %[[V_T3I3]], %[[T_ST]] addrspace(4)** %iiii, align 8, !tbaa !5

  union un u1 = {1};
  union un u2, *u3;
  u2 = __builtin_intel_fpga_reg(u1);
// CHECK: %[[V_TU1:[0-9]+]] = bitcast %[[T_UN]]* %agg-temp4 to i8*
// CHECK-NEXT: %[[V_TU2:[0-9]+]] = bitcast %[[T_UN]]* %u1 to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TU1]], i8* align 4 %[[V_TU2]], i64 4, i1 false), !tbaa.struct !14
// CHECK-NEXT: %[[V_TU3:[0-9]+]] = bitcast %[[T_UN]]* %agg-temp4 to i8*
// CHECK-NEXT: %[[V_TU4:[0-9]+]] = call i8* @llvm.ptr.annotation.p0i8(i8* %[[V_TU3]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_TU5:[0-9]+]] = bitcast i8* %[[V_TU4]] to %[[T_UN]]*
// CHECK-NEXT: %[[V_TU6:[0-9]+]] = bitcast %[[T_UN]]* %ref.tmp3 to i8*
// CHECK-NEXT: %[[V_TU7:[0-9]+]] = bitcast %[[T_UN]]* %[[V_TU5]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TU6]], i8* align 4 %[[V_TU7]], i64 8, i1 false)
// CHECK-NEXT: %[[V_TU8:[0-9]+]] = bitcast %[[T_UN]]* %u2 to i8*
// CHECK-NEXT: %[[V_TU9:[0-9]+]] = bitcast %[[T_UN]]* %ref.tmp3 to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TU8]], i8* align 4 %[[V_TU9]], i64 4, i1 false), !tbaa.struct !14

  u3 = __builtin_intel_fpga_reg(&u2);
// CHECK:      %[[V_TPU1:[0-9]+]] = ptrtoint %[[T_UN]]* %u2 to i64
// CHECK-NEXT: %[[V_TPU2:[0-9]+]] = call i64 @llvm.annotation.i64(i64 %[[V_TPU1]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_TPU3:[0-9]+]] = inttoptr i64 %[[V_TPU2]] to %[[T_UN]]*
// CHECK-NEXT: %[[V_TPU4:[0-9]+]] = addrspacecast %[[T_UN]]* %[[V_TPU3]] to %[[T_UN]] addrspace(4)*
// CHECK-NEXT: store %[[T_UN]] addrspace(4)* %[[V_TPU4]], %[[T_UN]] addrspace(4)** %u3, align 8, !tbaa !5

  A ca(213);
  A cb = __builtin_intel_fpga_reg(ca);
// CHECK: %[[V_TCA1:[0-9]+]] = bitcast %[[T_CL]]* %agg-temp5 to i8*
// CHECK-NEXT: %[[V_CA:[0-9]+]] = bitcast %[[T_CL]]* %ca to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_TCA1]], i8* align 4 %[[V_CA]], i64 4, i1 false), !tbaa.struct !16
// CHECK-NEXT: %[[V_TCA2:[0-9]+]] = bitcast %[[T_CL]]* %agg-temp5 to i8*
// CHECK-NEXT: %[[V_TCA3:[0-9]+]] = call i8* @llvm.ptr.annotation.p0i8(i8* %[[V_TCA2]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_TCA4:[0-9]+]] = bitcast i8* %[[V_TCA3]] to %[[T_CL]]*
// CHECK-NEXT: %[[V_CB:[0-9]+]] = bitcast %[[T_CL]]* %cb to i8*
// CHECK-NEXT: %[[V_TCA5:[0-9]+]] = bitcast %[[T_CL]]* %[[V_TCA4]] to i8*
// CHECK-NEXT: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %[[V_CB]], i8* align 4 %[[V_TCA5]], i64 8, i1 false)

  int *ap = &a;
  int *bp = __builtin_intel_fpga_reg(ap);
// CHECK: %[[V_AP0:[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)** %ap, align 8, !tbaa !5
// CHECK-NEXT: %[[V_AP1:[0-9]+]] = ptrtoint i32 addrspace(4)* %[[V_AP0]] to i64
// CHECK-NEXT: %[[V_AP2:[0-9]+]] = call i64 @llvm.annotation.i64(i64 %[[V_AP1]], [[BIFR_STR]]
// CHECK-NEXT: %[[V_AP3:[0-9]+]] = inttoptr i64 %[[V_AP2]] to i32 addrspace(4)*
// CHECK-NEXT: store i32 addrspace(4)* %[[V_AP3]], i32 addrspace(4)** %bp, align 8, !tbaa !5
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() { foo(); });
  return 0;
}

