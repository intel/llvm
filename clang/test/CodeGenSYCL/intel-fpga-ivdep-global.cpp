// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// Global ivdep - annotate all GEPs
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_no_paramv()
void ivdep_no_param() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_NO_PARAM:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_NO_PARAM:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_NO_PARAM:[0-9]+]]
  }
}

// Global ivdep - annotate all GEPs
// Make sure that ALL of the relevant GEPs for an array are marked into the array's index groups
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_no_param_multiple_gepsv()
void ivdep_no_param_multiple_geps() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  // CHECK: %[[TMP:[0-9a-z]+]] = alloca i32
  int t;
  [[intel::ivdep]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_MUL_GEPS:[0-9]+]]
    t = a[i];
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_MUL_GEPS:[0-9]+]]
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_MUL_GEPS]]
    a[i] = b[i];
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_MUL_GEPS]]
    b[i] = t;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_MUL_GEPS:[0-9]+]]
  }
}

// Global ivdep w/ safelen specified - annotate all GEPs
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_safelenv()
void ivdep_safelen() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep(5)]] for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_SAFELEN:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_SAFELEN:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_SAFELEN:[0-9]+]]
  }
}

// Global ivdep, albeit conflicting safelens - annotate all GEPs
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_conflicting_safelenv()
void ivdep_conflicting_safelen() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep(5)]]
  [[intel::ivdep(4)]] for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_CONFL_SAFELEN:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_CONFL_SAFELEN:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_CONFL_SAFELEN:[0-9]+]]
  }
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    ivdep_no_param();
    ivdep_no_param_multiple_geps();
    ivdep_safelen();
    ivdep_conflicting_safelen();
  });
  return 0;
}

// Find recurring instances of legacy "IVDep enable/safelen" MD nodes.
// CHECK-DAG: ![[IVDEP_LEGACY_ENABLE:[0-9]+]] = !{!"llvm.loop.ivdep.enable"}
// CHECK-DAG: ![[IVDEP_LEGACY_SAFELEN_5:[0-9]+]] = !{!"llvm.loop.ivdep.safelen", i32 5}

/// Global ivdep w/o safelen specified
/// All arrays have the same INF safelen - put access groups into the same parallel_access_indices metadata
//
// CHECK-DAG: ![[IDX_GROUP_A_NO_PARAM]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_NO_PARAM]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_NO_PARAM]] = distinct !{![[MD_LOOP_NO_PARAM]], ![[#]], ![[IVDEP_NO_PARAM:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_NO_PARAM]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_NO_PARAM]], ![[IDX_GROUP_B_NO_PARAM]]}
//
// CHECK-DAG: ![[IDX_GROUP_A_MUL_GEPS]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_MUL_GEPS]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_MUL_GEPS]] = distinct !{![[MD_LOOP_MUL_GEPS]], ![[#]], ![[IVDEP_MUL_GEPS:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_MUL_GEPS]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_MUL_GEPS]], ![[IDX_GROUP_B_MUL_GEPS]]}

/// Global ivdep w/ safelen specified
/// All arrays share the same safelen - put index groups into the same parallel_access_indices MD node
//
// CHECK-DAG: ![[IDX_GROUP_A_SAFELEN]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_SAFELEN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_SAFELEN]] = distinct !{![[MD_LOOP_SAFELEN]], ![[#]], ![[IVDEP_SAFELEN:[0-9]+]], ![[IVDEP_LEGACY_SAFELEN_5]]}
// CHECK-DAG: ![[IVDEP_SAFELEN]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_SAFELEN]], ![[IDX_GROUP_B_SAFELEN]], i32 5}

/// Conflicting global ivdeps, different safelens specified
/// The highest safelen must be used for all arrays
//
// CHECK-DAG: ![[IDX_GROUP_A_CONFL_SAFELEN]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_CONFL_SAFELEN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_CONFL_SAFELEN]] = distinct !{![[MD_LOOP_CONFL_SAFELEN]], ![[#]], ![[IVDEP_CONFL_SAFELEN:[0-9]+]], ![[IVDEP_LEGACY_SAFELEN_5]]}
// CHECK-DAG: ![[IVDEP_CONFL_SAFELEN]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_CONFL_SAFELEN]], ![[IDX_GROUP_B_CONFL_SAFELEN]], i32 5}
