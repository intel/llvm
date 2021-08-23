// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

struct st {
  int a;
  float b;
  char c;
};
// CHECK: [[T_ST:%struct[a-zA-Z0-9_.]*.st]] = type { i32, float, i8 }

union un {
  int a;
  char c[4];
};
// CHECK: [[T_UN:%union[a-zA-Z0-9_.]*.un]] = type { i32 }

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
// CHECK: [[T_CL:%class[a-zA-Z0-9_.]*.A]] = type { i32 }

// CHECK: @.str = private unnamed_addr constant [25 x i8] c"__builtin_intel_fpga_reg\00", section "llvm.metadata"

void scalars() {
  int a = 123;
  int b = __builtin_intel_fpga_reg(a);
  // CHECK: [[V_A1:%.*]] = load i32, i32 addrspace(4)* %a
  // CHECK-NEXT: [[V_A2:%.*]] = call i32 @llvm.annotation.i32(i32 [[V_A1]], [[BIFR_STR:i8\* getelementptr inbounds \(\[25 x i8\], \[25 x i8\]\* @.str, i32 0, i32 0\),]]
  // CHECK-NEXT: store i32 [[V_A2]], i32 addrspace(4)* %b

  int c = __builtin_intel_fpga_reg(2.0f);
  // CHECK: [[V_CF1:%.*]] = call i32 @llvm.annotation.i32(i32 1073741824, [[BIFR_STR]]
  // CHECK-NEXT: [[V_CF_BC:%.*]] = bitcast i32 [[V_CF1]] to float
  // CHECK-NEXT: [[V_CF2:%.*]] = fptosi float [[V_CF_BC]] to i32
  // CHECK-NEXT: store i32 [[V_CF2]], i32 addrspace(4)* %c

  int d = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( b+12 ));
  // CHECK: [[V_B1:%.*]] = load i32, i32 addrspace(4)* %b
  // CHECK-NEXT: [[V_B2:%.*]] = add nsw i32 [[V_B1]], 12
  // CHECK-NEXT: [[V_B3:%.*]] = call i32 @llvm.annotation.i32(i32 [[V_B2]], [[BIFR_STR]]
  // CHECK-NEXT: [[V_B4:%.*]] = call i32 @llvm.annotation.i32(i32 [[V_B3]], [[BIFR_STR]]
  // CHECK-NEXT: store i32 [[V_B4]], i32 addrspace(4)* %d

  int e = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( a+b ));
  // CHECK: [[V_AB1:%.*]] = load i32, i32 addrspace(4)* %a
  // CHECK-NEXT: [[V_AB2:%.*]] = load i32, i32 addrspace(4)* %b
  // CHECK-NEXT: [[V_AB3:%.*]] = add nsw i32 [[V_AB1]], [[V_AB2]]
  // CHECK-NEXT: [[V_AB4:%.*]] = call i32 @llvm.annotation.i32(i32 [[V_AB3]], [[BIFR_STR]]
  // CHECK-NEXT: [[V_AB5:%.*]] = call i32 @llvm.annotation.i32(i32 [[V_AB4]], [[BIFR_STR]]
  // CHECK-NEXT: store i32 [[V_AB5]], i32 addrspace(4)* %e

  int f;
  f = __builtin_intel_fpga_reg(a);
  // CHECK: [[V_F1:%.*]] = load i32, i32 addrspace(4)* %a
  // CHECK-NEXT: [[V_F2:%.*]] = call i32 @llvm.annotation.i32(i32 [[V_F1]], [[BIFR_STR]]
  // CHECK-NEXT: store i32 [[V_F2]], i32 addrspace(4)* %f
}

void structs() {
  // CHECK: [[S1:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[S1_ASCAST:%.*]] = addrspacecast [[T_ST]]* [[S1]] to [[T_ST]] addrspace(4)*
  // CHECK-NEXT: [[S2:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[S2_ASCAST:%.*]] = addrspacecast [[T_ST]]* [[S2]] to [[T_ST]] addrspace(4)*
  // CHECK-NEXT: [[S3:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[S3_ASCAST:%.*]] = addrspacecast [[T_ST]]* [[S3]] to [[T_ST]] addrspace(4)*
  // CHECK-NEXT: [[REF_TMP:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[REF_TMP_ASCAST:%.*]] = addrspacecast [[T_ST]]* [[REF_TMP]] to [[T_ST]] addrspace(4)*
  struct st s1;

  struct st s2 = __builtin_intel_fpga_reg(s1);
  // CHECK: [[TMP_S1:%.*]] = bitcast [[T_ST]] addrspace(4)* [[S2_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_S2:%.*]] = bitcast [[T_ST]] addrspace(4)* [[S1_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[TMP_S1]], i8 addrspace(4)* align 4 [[TMP_S2]], i64 12, i1 false)
  // CHECK-NEXT: [[TMP_S3:%.*]] = bitcast [[T_ST]] addrspace(4)* [[S2_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_S4:%.*]] = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)* [[TMP_S3]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_S5:%.*]] = bitcast i8 addrspace(4)* [[TMP_S4]] to [[T_ST]] addrspace(4)*

  struct st s3;
  s3 = __builtin_intel_fpga_reg(s2);
  // CHECK: [[TMP_S6:%.*]] = bitcast [[T_ST]] addrspace(4)* [[REF_TMP_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_S7:%.*]] = bitcast [[T_ST]] addrspace(4)* [[S2_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[TMP_S6]], i8 addrspace(4)* align 4 [[TMP_S7]], i64 12, i1 false)
  // CHECK-NEXT: [[TMP_S8:%.*]] = bitcast [[T_ST]] addrspace(4)* [[REF_TMP_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_S9:%.*]] = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)* [[TMP_S8]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_S10:%.*]] = bitcast i8 addrspace(4)* [[TMP_S9]] to [[T_ST]] addrspace(4)*
  // CHECK-NEXT: [[TMP_S11:%.*]] = bitcast [[T_ST]] addrspace(4)* [[S3_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_S12:%.*]] = bitcast [[T_ST]] addrspace(4)* [[REF_TMP_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[TMP_S11]], i8 addrspace(4)* align 4 [[TMP_S12]], i64 12, i1 false)
}

void unions() {
  // CHECK: [[U1:%.*]] = alloca [[T_UN]], align 4
  // CHECK-NEXT: [[U1_ASCAST:%.*]] = addrspacecast [[T_UN]]* [[U1]] to [[T_UN]] addrspace(4)*
  // CHECK-NEXT: [[U2:%.*]] = alloca [[T_UN]], align 4
  // CHECK-NEXT: [[U2_ASCAST:%.*]] = addrspacecast [[T_UN]]* [[U2]] to [[T_UN]] addrspace(4)*
  // CHECK-NEXT: [[REF_TMP2:%.*]] = alloca [[T_UN]], align 4
  // CHECK-NEXT: [[REF_TMP2_ASCAST:%.*]] = addrspacecast [[T_UN]]* [[REF_TMP2]] to [[T_UN]] addrspace(4)*
  union un u1;
  union un u2;

  u2 = __builtin_intel_fpga_reg(u1);
  // CHECK: [[TMP_U1:%.*]] = bitcast [[T_UN]] addrspace(4)* [[REF_TMP2_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_U2:%.*]] = bitcast [[T_UN]] addrspace(4)* [[U1_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[TMP_U1]], i8 addrspace(4)* align 4 [[TMP_U2]], i64 4, i1 false)
  // CHECK-NEXT: [[TMP_U3:%.*]] = bitcast [[T_UN]] addrspace(4)* [[REF_TMP2_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_U4:%.*]] = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)* [[TMP_U3]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_U5:%.*]] = bitcast i8 addrspace(4)* [[TMP_U4]] to [[T_UN]] addrspace(4)*
  // CHECK-NEXT: [[TMP_U6:%.*]] = bitcast [[T_UN]] addrspace(4)* [[U2_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_U7:%.*]] = bitcast [[T_UN]] addrspace(4)* [[REF_TMP2_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[TMP_U6]], i8 addrspace(4)* align 4 [[TMP_U7]], i64 4, i1 false)
}

void classes() {
  // CHECK: [[CA:%.*]] = alloca [[T_CL:%.*]], align 4
  // CHECK-NEXT: [[CA_ASCAST:%.*]] = addrspacecast [[T_CL]]* [[CA]] to [[T_CL]] addrspace(4)*
  // CHECK-NEXT: [[CB:%.*]] = alloca [[T_CL]], align 4
  // CHECK-NEXT: [[CB_ASCAST:%.*]] = addrspacecast [[T_CL]]* [[CB]] to [[T_CL]] addrspace(4)*
  A ca(213);

  A cb = __builtin_intel_fpga_reg(ca);
  // CHECK: [[TMP_C1:%.*]] = bitcast [[T_CL]] addrspace(4)* [[CB_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_C2:%.*]] = bitcast [[T_CL]] addrspace(4)* [[CA_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 4 [[TMP_C1]], i8 addrspace(4)* align 4 [[TMP_C2]], i64 4, i1 false)
  // CHECK-NEXT: [[TMP_C3:%.*]] = bitcast [[T_CL]] addrspace(4)* [[CB_ASCAST]] to i8 addrspace(4)*
  // CHECK-NEXT: [[TMP_C4:%.*]] = call i8 addrspace(4)* @llvm.ptr.annotation.p4i8(i8 addrspace(4)* [[TMP_C3]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_C5:%.*]] = bitcast i8 addrspace(4)* [[TMP_C4]] to [[T_CL]] addrspace(4)*
}

void pointers() {
  int v;
  int *pv = &v;
  int *pv2 = __builtin_intel_fpga_reg(pv);
  // CHECK: [[TMP_P1:%[0-9]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %pv.ascast, align 8
  // CHECK-NEXT: [[TMP_P2:%[0-9]+]] = ptrtoint i32 addrspace(4)* [[TMP_P1]] to i64
  // CHECK-NEXT: [[TMP_P3:%[0-9]+]] = call i64 @llvm.annotation.i64(i64 [[TMP_P2]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_P4:%[0-9]+]] = inttoptr i64 [[TMP_P3]] to i32 addrspace(4)*
  // CHECK-NEXT: store i32 addrspace(4)* [[TMP_P4]], i32 addrspace(4)* addrspace(4)* %pv2.ascast, align 8

  struct st s;
  struct st *ps = __builtin_intel_fpga_reg(&s);
  // CHECK: [[TMP_P5:%.*]] = ptrtoint [[T_ST]] addrspace(4)* %s.ascast to i64
  // CHECK-NEXT: [[TMP_P6:%.*]] = call i64 @llvm.annotation.i64(i64 [[TMP_P5]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_P7:%.*]] = inttoptr i64 [[TMP_P6]] to [[T_ST]] addrspace(4)*
  // CHECK-NEXT: store [[T_ST]] addrspace(4)* [[TMP_P7]], [[T_ST]] addrspace(4)* addrspace(4)* %ps.ascast, align 8

  union un u, *pu;
  pu = __builtin_intel_fpga_reg(&u);
  // CHECK: [[TMP_P8:%.*]] = ptrtoint [[T_UN]] addrspace(4)* %u.ascast to i64
  // CHECK-NEXT: [[TMP_P9:%.*]] = call i64 @llvm.annotation.i64(i64 [[TMP_P8]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_P10:%.*]] = inttoptr i64 [[TMP_P9]] to [[T_UN]] addrspace(4)*
  // CHECK-NEXT: store [[T_UN]] addrspace(4)* [[TMP_P10]], [[T_UN]] addrspace(4)* addrspace(4)* %pu.ascast, align 8
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class fake_kernel>([]() {
    scalars();
    structs();
    unions();
    classes();
    pointers();
  });
  return 0;
}

