// RUN: %clang_cc1 -x c++ -triple spir64-unknown-linux-sycldevice -std=c++11 -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// Array-specific ivdep - annotate the correspondent GEPs only
//
// CHECK: define spir_func void @_Z{{[0-9]+}}ivdep_array_no_safelenv()
void ivdep_array_no_safelen() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intelfpga::ivdep(a)]]
  for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_A]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_ARR:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_B]], i64 0, i64 %{{[0-9a-z]+}}{{[[:space:]]}}
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR:[0-9]+]]
  }
}

// Array-specific ivdep w/ safelen - annotate the correspondent GEPs only
// CHECK: define spir_func void @_Z{{[0-9]+}}ivdep_array_with_safelenv()
void ivdep_array_with_safelen() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intelfpga::ivdep(a, 5)]]
  for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_A]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_ARR_SAFELEN:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_B]], i64 0, i64 %{{[0-9a-z]+}}{{[[:space:]]}}
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_SAFELEN:[0-9]+]]
  }
}

// Multiple array-specific ivdeps - annotate the correspondent GEPs
//
// CHECK: define spir_func void @_Z{{[0-9]+}}ivdep_multiple_arraysv()
void ivdep_multiple_arrays() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  // CHECK: %[[ARRAY_C:[0-9a-z]+]] = alloca [10 x i32]
  int c[10];
  // CHECK: %[[ARRAY_D:[0-9a-z]+]] = alloca [10 x i32]
  int d[10];
  [[intelfpga::ivdep(a, 5)]]
  [[intelfpga::ivdep(b, 5)]]
  [[intelfpga::ivdep(c)]]
  [[intelfpga::ivdep(d)]]
  for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_A]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_MUL_ARR:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_B]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_MUL_ARR:[0-9]+]]
    b[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_C]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_C_MUL_ARR:[0-9]+]]
    c[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_D]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_D_MUL_ARR:[0-9]+]]
    d[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_MUL_ARR:[0-9]+]]
  }
}

// Global ivdep with INF safelen & array-specific ivdep with the same safelen
//
// CHECK: define spir_func void @_Z{{[0-9]+}}ivdep_array_and_globalv()
void ivdep_array_and_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intelfpga::ivdep]]
  [[intelfpga::ivdep(a)]]
  for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_A]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_ARR_AND_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_B]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_ARR_AND_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_AND_GLOB:[0-9]+]]
  }
}

// Global ivdep with INF safelen & array-specific ivdep with lesser safelen
//
// CHECK: define spir_func void @_Z{{[0-9]+}}ivdep_array_and_inf_globalv()
void ivdep_array_and_inf_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intelfpga::ivdep]]
  [[intelfpga::ivdep(a, 8)]]
  for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_A]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_ARR_AND_INF_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_B]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_ARR_AND_INF_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_AND_INF_GLOB:[0-9]+]]
  }
}

// Global ivdep with specified safelen & array-specific ivdep with lesser safelen
//
// CHECK: define spir_func void @_Z{{[0-9]+}}ivdep_array_and_greater_globalv()
void ivdep_array_and_greater_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intelfpga::ivdep(9)]]
  [[intelfpga::ivdep(a, 8)]]
  for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_A]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_ARR_AND_GREAT_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_B]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_ARR_AND_GREAT_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_AND_GREAT_GLOB:[0-9]+]]
  }
}

// Global safelen, array-specific safelens
//
// CHECK: define spir_func void @_Z{{[0-9]+}}ivdep_mul_arrays_and_globalv()
void ivdep_mul_arrays_and_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  // CHECK: %[[ARRAY_C:[0-9a-z]+]] = alloca [10 x i32]
  int c[10];
  [[intelfpga::ivdep(5)]]
  [[intelfpga::ivdep(b, 6)]]
  [[intelfpga::ivdep(c)]]
  for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_A]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_MUL_ARR_AND_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_B]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_MUL_ARR_AND_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32]* %[[ARRAY_C]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_C_MUL_ARR_AND_GLOB:[0-9]+]]
    c[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_MUL_ARR_AND_GLOB:[0-9]+]]
  }
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(Func kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    ivdep_array_no_safelen();
    ivdep_array_with_safelen();
    ivdep_multiple_arrays();
    ivdep_array_and_global();
    ivdep_array_and_inf_global();
    ivdep_array_and_greater_global();
    ivdep_mul_arrays_and_global();
  });
  return 0;
}

/// A particular array with no safelen specified
//
// CHECK-DAG: ![[IDX_GROUP_ARR]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR]] = distinct !{![[MD_LOOP_ARR]], ![[IVDEP_ARR:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_ARR]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_ARR]]}

/// A particular array with safelen specified
//
// CHECK: ![[IDX_GROUP_ARR_SAFELEN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_SAFELEN]] = distinct !{![[MD_LOOP_ARR_SAFELEN]], ![[IVDEP_ARR_SAFELEN:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_ARR_SAFELEN]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_ARR_SAFELEN]], i32 5}

/// Multiple arrays.
/// Index groups for arrays with matching safelens should be put into the same parallel_access_indices MD node
//
// CHECK-DAG: ![[IDX_GROUP_A_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_C_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_D_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_MUL_ARR]] = distinct !{![[MD_LOOP_MUL_ARR]], ![[IVDEP_MUL_ARR_VAL:[0-9]+]], ![[IVDEP_MUL_ARR_INF:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_MUL_ARR_VAL]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_MUL_ARR]], ![[IDX_GROUP_B_MUL_ARR]], i32 5}
// CHECK-DAG: ![[IVDEP_MUL_ARR_INF]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_C_MUL_ARR]], ![[IDX_GROUP_D_MUL_ARR]]}

/// Global INF safelen and specific array INF safelen
/// The array-specific ivdep can be ignored, so it's the same as just global ivdep with safelen INF
//
// CHECK-DAG: ![[IDX_GROUP_A_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_AND_GLOB]] = distinct !{![[MD_LOOP_ARR_AND_GLOB]], ![[IVDEP_ARR_AND_GLOB:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_ARR_AND_GLOB]], ![[IDX_GROUP_B_ARR_AND_GLOB]]}

/// Global INF safelen and specific array non-INF safelen
/// The array-specific ivdep must be ignored, so it's the same as just global ivdep with safelen INF
//
// CHECK-DAG: ![[IDX_GROUP_A_ARR_AND_INF_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_ARR_AND_INF_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_AND_INF_GLOB]] = distinct !{![[MD_LOOP_ARR_AND_INF_GLOB]], ![[IVDEP_ARR_AND_INF_GLOB:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_ARR_AND_INF_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_ARR_AND_INF_GLOB]], ![[IDX_GROUP_B_ARR_AND_INF_GLOB]]}

/// Global safelen and specific array with lesser safelen
/// The array-specific ivdep must be gnored, so it's the same as just global ivdep with its safelen
//
// CHECK-DAG: ![[IDX_GROUP_A_ARR_AND_GREAT_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_ARR_AND_GREAT_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_AND_GREAT_GLOB]] = distinct !{![[MD_LOOP_ARR_AND_GREAT_GLOB]], ![[IVDEP_ARR_AND_GREAT_GLOB:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_ARR_AND_GREAT_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_ARR_AND_GREAT_GLOB]], ![[IDX_GROUP_B_ARR_AND_GREAT_GLOB]], i32 9}

/// Multiple arrays with specific safelens and lesser global safelen
/// The array-specific safelens are kept for the correspondent arrays, the global safelen applies to the rest
//
// CHECK-DAG: ![[IDX_GROUP_A_MUL_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_MUL_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_C_MUL_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_MUL_ARR_AND_GLOB]] = distinct !{![[MD_LOOP_MUL_ARR_AND_GLOB]], ![[IVDEP_A_MUL_ARR_AND_GLOB:[0-9]+]], ![[IVDEP_B_MUL_ARR_AND_GLOB:[0-9]+]], ![[IVDEP_C_MUL_ARR_AND_GLOB:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_A_MUL_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_MUL_ARR_AND_GLOB]], i32 5}
// CHECK-DAG: ![[IVDEP_B_MUL_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_B_MUL_ARR_AND_GLOB]], i32 6}
// CHECK-DAG: ![[IVDEP_C_MUL_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_C_MUL_ARR_AND_GLOB]]}
//
