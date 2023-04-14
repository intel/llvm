// RUN: %clang_cc1 -fsycl-is-device -triple spir64-unknown-unknown -disable-llvm-passes -opaque-pointers -emit-llvm %s -o - | FileCheck %s

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

// CHECK: @.str = private unnamed_addr addrspace(1) constant [25 x i8] c"__builtin_intel_fpga_reg\00", section "llvm.metadata"

void scalars() {
  int a = 123;
  int b = __builtin_intel_fpga_reg(a);
  // CHECK: [[V_A1:%.*]] = load i32, ptr addrspace(4) %a
  // CHECK-NEXT: [[V_A2:%.*]] = call i32 @llvm.annotation.i32.p1(i32 [[V_A1]], [[BIFR_STR:ptr addrspace\(1\) @.str,]]
  // CHECK-NEXT: store i32 [[V_A2]], ptr addrspace(4) %b

  int c = __builtin_intel_fpga_reg(2.0f);
  // CHECK: [[V_CF1:%.*]] = call i32 @llvm.annotation.i32.p1(i32 1073741824, [[BIFR_STR]]
  // CHECK-NEXT: [[V_CF_BC:%.*]] = bitcast i32 [[V_CF1]] to float
  // CHECK-NEXT: [[V_CF2:%.*]] = fptosi float [[V_CF_BC]] to i32
  // CHECK-NEXT: store i32 [[V_CF2]], ptr addrspace(4) %c

  int d = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( b+12 ));
  // CHECK: [[V_B1:%.*]] = load i32, ptr addrspace(4) %b
  // CHECK-NEXT: [[V_B2:%.*]] = add nsw i32 [[V_B1]], 12
  // CHECK-NEXT: [[V_B3:%.*]] = call i32 @llvm.annotation.i32.p1(i32 [[V_B2]], [[BIFR_STR]]
  // CHECK-NEXT: [[V_B4:%.*]] = call i32 @llvm.annotation.i32.p1(i32 [[V_B3]], [[BIFR_STR]]
  // CHECK-NEXT: store i32 [[V_B4]], ptr addrspace(4) %d

  int e = __builtin_intel_fpga_reg( __builtin_intel_fpga_reg( a+b ));
  // CHECK: [[V_AB1:%.*]] = load i32, ptr addrspace(4) %a
  // CHECK-NEXT: [[V_AB2:%.*]] = load i32, ptr addrspace(4) %b
  // CHECK-NEXT: [[V_AB3:%.*]] = add nsw i32 [[V_AB1]], [[V_AB2]]
  // CHECK-NEXT: [[V_AB4:%.*]] = call i32 @llvm.annotation.i32.p1(i32 [[V_AB3]], [[BIFR_STR]]
  // CHECK-NEXT: [[V_AB5:%.*]] = call i32 @llvm.annotation.i32.p1(i32 [[V_AB4]], [[BIFR_STR]]
  // CHECK-NEXT: store i32 [[V_AB5]], ptr addrspace(4) %e

  int f;
  f = __builtin_intel_fpga_reg(a);
  // CHECK: [[V_F1:%.*]] = load i32, ptr addrspace(4) %a
  // CHECK-NEXT: [[V_F2:%.*]] = call i32 @llvm.annotation.i32.p1(i32 [[V_F1]], [[BIFR_STR]]
  // CHECK-NEXT: store i32 [[V_F2]], ptr addrspace(4) %f
}

void structs() {
  // CHECK: [[S1:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[S2:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[S3:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[REF_TMP:%.*]] = alloca [[T_ST]], align 4
  // CHECK-NEXT: [[S1_ASCAST:%.*]] = addrspacecast ptr [[S1]] to ptr addrspace(4)
  // CHECK-NEXT: [[S2_ASCAST:%.*]] = addrspacecast ptr [[S2]] to ptr addrspace(4)
  // CHECK-NEXT: [[S3_ASCAST:%.*]] = addrspacecast ptr [[S3]] to ptr addrspace(4)
  // CHECK-NEXT: [[REF_TMP_ASCAST:%.*]] = addrspacecast ptr [[REF_TMP]] to ptr addrspace(4)
  struct st s1;

  struct st s2 = __builtin_intel_fpga_reg(s1);
  // CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[S2_ASCAST]], ptr addrspace(4) align 4 [[S1_ASCAST]], i64 12, i1 false)
  // CHECK: [[TMP_S4:%.*]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) [[S2_ASCAST]], [[BIFR_STR]]

  struct st s3;
  s3 = __builtin_intel_fpga_reg(s2);
  // CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[REF_TMP_ASCAST]], ptr addrspace(4) align 4 [[S2_ASCAST]], i64 12, i1 false)
  // CHECK: [[TMP_S9:%.*]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) [[REF_TMP_ASCAST]], [[BIFR_STR]]
  // CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[S3_ASCAST]], ptr addrspace(4) align 4 [[REF_TMP_ASCAST]], i64 12, i1 false)
}

void unions() {
  // CHECK: [[U1:%.*]] = alloca [[T_UN]], align 4
  // CHECK-NEXT: [[U2:%.*]] = alloca [[T_UN]], align 4
  // CHECK-NEXT: [[REF_TMP2:%.*]] = alloca [[T_UN]], align 4
  // CHECK-NEXT: [[U1_ASCAST:%.*]] = addrspacecast ptr [[U1]] to ptr addrspace(4)
  // CHECK-NEXT: [[U2_ASCAST:%.*]] = addrspacecast ptr [[U2]] to ptr addrspace(4)
  // CHECK-NEXT: [[REF_TMP2_ASCAST:%.*]] = addrspacecast ptr [[REF_TMP2]] to ptr addrspace(4)
  union un u1;
  union un u2;

  u2 = __builtin_intel_fpga_reg(u1);
  // CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[REF_TMP2_ASCAST]], ptr addrspace(4) align 4 [[U1_ASCAST]], i64 4, i1 false)
  // CHECK-NEXT: [[TMP_U4:%.*]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) [[REF_TMP2_ASCAST]], [[BIFR_STR]]
  // CHECK-NEXT: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[U2_ASCAST]], ptr addrspace(4) align 4 [[REF_TMP2_ASCAST]], i64 4, i1 false)
}

void classes() {
  // CHECK: [[CA:%.*]] = alloca [[T_CL:%.*]], align 4
  // CHECK-NEXT: [[CB:%.*]] = alloca [[T_CL]], align 4
  // CHECK-NEXT: [[CA_ASCAST:%.*]] = addrspacecast ptr [[CA]] to ptr addrspace(4)
  // CHECK-NEXT: [[CB_ASCAST:%.*]] = addrspacecast ptr [[CB]] to ptr addrspace(4)
  A ca(213);

  A cb = __builtin_intel_fpga_reg(ca);
  // CHECK: call void @llvm.memcpy.p4.p4.i64(ptr addrspace(4) align 4 [[CB_ASCAST]], ptr addrspace(4) align 4 [[CA_ASCAST]], i64 4, i1 false)
  // CHECK-NEXT: [[TMP_C4:%.*]] = call ptr addrspace(4) @llvm.ptr.annotation.p4.p1(ptr addrspace(4) [[CB_ASCAST]], [[BIFR_STR]]
}

void pointers() {
  int v;
  int *pv = &v;
  int *pv2 = __builtin_intel_fpga_reg(pv);
  // CHECK: [[TMP_P1:%[0-9]+]] = load ptr addrspace(4), ptr addrspace(4) %pv.ascast, align 8
  // CHECK-NEXT: [[TMP_P2:%[0-9]+]] = ptrtoint ptr addrspace(4) [[TMP_P1]] to i64
  // CHECK-NEXT: [[TMP_P3:%[0-9]+]] = call i64 @llvm.annotation.i64.p1(i64 [[TMP_P2]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_P4:%[0-9]+]] = inttoptr i64 [[TMP_P3]] to ptr addrspace(4)
  // CHECK-NEXT: store ptr addrspace(4) [[TMP_P4]], ptr addrspace(4) %pv2.ascast, align 8

  struct st s;
  struct st *ps = __builtin_intel_fpga_reg(&s);
  // CHECK: [[TMP_P5:%.*]] = ptrtoint ptr addrspace(4) %s.ascast to i64
  // CHECK-NEXT: [[TMP_P6:%.*]] = call i64 @llvm.annotation.i64.p1(i64 [[TMP_P5]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_P7:%.*]] = inttoptr i64 [[TMP_P6]] to ptr addrspace(4)
  // CHECK-NEXT: store ptr addrspace(4) [[TMP_P7]], ptr addrspace(4) %ps.ascast, align 8

  union un u, *pu;
  pu = __builtin_intel_fpga_reg(&u);
  // CHECK: [[TMP_P8:%.*]] = ptrtoint ptr addrspace(4) %u.ascast to i64
  // CHECK-NEXT: [[TMP_P9:%.*]] = call i64 @llvm.annotation.i64.p1(i64 [[TMP_P8]], [[BIFR_STR]]
  // CHECK-NEXT: [[TMP_P10:%.*]] = inttoptr i64 [[TMP_P9]] to ptr addrspace(4)
  // CHECK-NEXT: store ptr addrspace(4) [[TMP_P10]], ptr addrspace(4) %pu.ascast, align 8
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

