// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// Array-specific ivdep - annotate the correspondent GEPs only
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_array_no_safelenv()
void ivdep_array_no_safelen() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep(a)]] for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_ARR:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}{{[[:space:]]}}
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR:[0-9]+]]
  }
}

// Array-specific ivdep w/ safelen - annotate the correspondent GEPs only
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_array_with_safelenv()
void ivdep_array_with_safelen() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep(a, 5)]] for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_ARR_SAFELEN:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}{{[[:space:]]}}
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_SAFELEN:[0-9]+]]
  }
}

// Multiple array-specific ivdeps - annotate the correspondent GEPs
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_multiple_arraysv()
void ivdep_multiple_arrays() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  // CHECK: %[[ARRAY_C:[0-9a-z]+]] = alloca [10 x i32]
  int c[10];
  // CHECK: %[[ARRAY_D:[0-9a-z]+]] = alloca [10 x i32]
  int d[10];
  [[intel::ivdep(a, 5)]]
  [[intel::ivdep(b, 5)]]
  [[intel::ivdep(c)]]
  [[intel::ivdep(d)]] for (int i = 0; i != 10; ++i) {
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_MUL_ARR:[0-9]+]]
    a[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_MUL_ARR:[0-9]+]]
    b[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_C]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_C_MUL_ARR:[0-9]+]]
    c[i] = 0;
    // CHECK:  %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_D]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_D_MUL_ARR:[0-9]+]]
    d[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_MUL_ARR:[0-9]+]]
  }
}

// Global ivdep with INF safelen & array-specific ivdep with the same safelen
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_array_and_globalv()
void ivdep_array_and_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep]]
  [[intel::ivdep(a)]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_ARR_AND_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_ARR_AND_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_AND_GLOB:[0-9]+]]
  }
}

// Global ivdep with INF safelen & array-specific ivdep with lesser safelen
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_array_and_inf_globalv()
void ivdep_array_and_inf_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep]]
  [[intel::ivdep(a, 8)]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_ARR_AND_INF_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_ARR_AND_INF_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_AND_INF_GLOB:[0-9]+]]
  }
}

// Global ivdep with specified safelen & array-specific ivdep with lesser safelen
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_array_and_greater_globalv()
void ivdep_array_and_greater_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep(9)]]
  [[intel::ivdep(a, 8)]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_ARR_AND_GREAT_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_ARR_AND_GREAT_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_ARR_AND_GREAT_GLOB:[0-9]+]]
  }
}

// Global safelen, array-specific safelens
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_mul_arrays_and_globalv()
void ivdep_mul_arrays_and_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  // CHECK: %[[ARRAY_C:[0-9a-z]+]] = alloca [10 x i32]
  int c[10];
  [[intel::ivdep(5)]]
  [[intel::ivdep(b, 6)]]
  [[intel::ivdep(c)]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_MUL_ARR_AND_GLOB:[0-9]+]]
    a[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_MUL_ARR_AND_GLOB:[0-9]+]]
    b[i] = 0;
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_C]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_C_MUL_ARR_AND_GLOB:[0-9]+]]
    c[i] = 0;
    // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_MUL_ARR_AND_GLOB:[0-9]+]]
  }
}

// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_ptrv()
void ivdep_ptr() {
  int *ptr;
  // CHECK: %[[PTR:[0-9a-z]+]] = alloca i32 addrspace(4)*
  [[intel::ivdep(ptr, 5)]] for (int i = 0; i != 10; ++i)
      ptr[i] = 0;
  // CHECK: %[[PTR_LOAD:[0-9a-z]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %[[PTR]]
  // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds i32, i32 addrspace(4)* %[[PTR_LOAD]], i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_PTR:[0-9]+]]
  // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_PTR:[0-9]+]]
}

// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_structv()
void ivdep_struct() {
  struct S {
    int *ptr;
    int arr[10];
  } s;
  // CHECK: %[[STRUCT:[0-9a-z]+]] = alloca %struct.{{.+}}.S
  [[intel::ivdep(s.arr, 5)]] for (int i = 0; i != 10; ++i)
      s.arr[i] = 0;
  // CHECK: %[[STRUCT_ARR:[0-9a-z]+]] = getelementptr inbounds %struct.{{.+}}.S, %struct.{{.+}}.S addrspace(4)* %[[STRUCT]].ascast, i32 0, i32 1
  // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[STRUCT_ARR]], i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_STRUCT_ARR:[0-9]+]]
  // CHECK: br label %for.cond, !llvm.loop ![[MD_LOOP_STRUCT_ARR:[0-9]+]]

  [[intel::ivdep(s.ptr, 5)]] for (int i = 0; i != 10; ++i)
      s.ptr[i] = 0;
  // CHECK: %[[STRUCT_PTR:[0-9a-z]+]] = getelementptr inbounds %struct.{{.+}}.S, %struct.{{.+}}.S addrspace(4)* %[[STRUCT]].ascast, i32 0, i32 0
  // CHECK: %[[LOAD_STRUCT_PTR:[0-9a-z]+]] = load i32 addrspace(4)*, i32 addrspace(4)* addrspace(4)* %[[STRUCT_PTR]]
  // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds i32, i32 addrspace(4)* %[[LOAD_STRUCT_PTR]], i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_STRUCT_PTR:[0-9]+]]
  // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_STRUCT_PTR:[0-9]+]]
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
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
    ivdep_ptr();
    ivdep_struct();
  });
  return 0;
}

/// A particular array with no safelen specified
//
// CHECK-DAG: ![[IDX_GROUP_ARR]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR]] = distinct !{![[MD_LOOP_ARR]], ![[MP:[0-9]+]], ![[IVDEP_ARR:[0-9]+]]}
// CHECK-DAG: ![[MP]] = !{!"llvm.loop.mustprogress"}
// CHECK-DAG: ![[IVDEP_ARR]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_ARR]]}

/// A particular array with safelen specified
//
// CHECK: ![[IDX_GROUP_ARR_SAFELEN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_SAFELEN]] = distinct !{![[MD_LOOP_ARR_SAFELEN]], ![[MP]], ![[IVDEP_ARR_SAFELEN:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_ARR_SAFELEN]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_ARR_SAFELEN]], i32 5}

/// Multiple arrays.
/// Index groups for arrays with matching safelens should be put into the same parallel_access_indices MD node
//
// CHECK-DAG: ![[IDX_GROUP_A_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_C_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_D_MUL_ARR]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_MUL_ARR]] = distinct !{![[MD_LOOP_MUL_ARR]], ![[MP]], ![[IVDEP_MUL_ARR_VAL:[0-9]+]], ![[IVDEP_MUL_ARR_INF:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_MUL_ARR_VAL]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_MUL_ARR]], ![[IDX_GROUP_B_MUL_ARR]], i32 5}
// CHECK-DAG: ![[IVDEP_MUL_ARR_INF]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_C_MUL_ARR]], ![[IDX_GROUP_D_MUL_ARR]]}

// Find the single instance of a legacy "IVDep enable" MD node.
// CHECK-DAG: ![[IVDEP_LEGACY_ENABLE:[0-9]+]] = !{!"llvm.loop.ivdep.enable"}

/// Global INF safelen and specific array INF safelen
/// The array-specific ivdep can be ignored, so it's the same as just global ivdep with safelen INF
//
// CHECK-DAG: ![[IDX_GROUP_A_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_AND_GLOB]] = distinct !{![[MD_LOOP_ARR_AND_GLOB]], ![[MP]], ![[IVDEP_ARR_AND_GLOB:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_ARR_AND_GLOB]], ![[IDX_GROUP_B_ARR_AND_GLOB]]}

/// Global INF safelen and specific array non-INF safelen
/// The array-specific ivdep must be ignored, so it's the same as just global ivdep with safelen INF
//
// CHECK-DAG: ![[IDX_GROUP_A_ARR_AND_INF_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_ARR_AND_INF_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_AND_INF_GLOB]] = distinct !{![[MD_LOOP_ARR_AND_INF_GLOB]], ![[MP]], ![[IVDEP_ARR_AND_INF_GLOB:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_ARR_AND_INF_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_ARR_AND_INF_GLOB]], ![[IDX_GROUP_B_ARR_AND_INF_GLOB]]}

/// Global safelen and specific array with lesser safelen
/// The array-specific ivdep must be gnored, so it's the same as just global ivdep with its safelen
//
// CHECK-DAG: ![[IDX_GROUP_A_ARR_AND_GREAT_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_ARR_AND_GREAT_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_ARR_AND_GREAT_GLOB]] = distinct !{![[MD_LOOP_ARR_AND_GREAT_GLOB]], ![[MP]], ![[IVDEP_ARR_AND_GREAT_GLOB:[0-9]+]], ![[IVDEP_LEGACY_ARR_AND_GREAT_GLOB:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_LEGACY_ARR_AND_GREAT_GLOB]] = !{!"llvm.loop.ivdep.safelen", i32 9}
// CHECK-DAG: ![[IVDEP_ARR_AND_GREAT_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_ARR_AND_GREAT_GLOB]], ![[IDX_GROUP_B_ARR_AND_GREAT_GLOB]], i32 9}

/// Multiple arrays with specific safelens and lesser global safelen
/// The array-specific safelens are kept for the correspondent arrays, the global safelen applies to the rest
//
// CHECK-DAG: ![[IDX_GROUP_A_MUL_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_MUL_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_C_MUL_ARR_AND_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_MUL_ARR_AND_GLOB]] = distinct !{![[MD_LOOP_MUL_ARR_AND_GLOB]], ![[MP]], ![[IVDEP_A_MUL_ARR_AND_GLOB:[0-9]+]], ![[IVDEP_LEGACY_MUL_ARR_AND_GLOB:[0-9]+]], ![[IVDEP_B_MUL_ARR_AND_GLOB:[0-9]+]], ![[IVDEP_C_MUL_ARR_AND_GLOB:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_LEGACY_MUL_ARR_AND_GLOB]] = !{!"llvm.loop.ivdep.safelen", i32 5}
// CHECK-DAG: ![[IVDEP_A_MUL_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_MUL_ARR_AND_GLOB]], i32 5}
// CHECK-DAG: ![[IVDEP_B_MUL_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_B_MUL_ARR_AND_GLOB]], i32 6}
// CHECK-DAG: ![[IVDEP_C_MUL_ARR_AND_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_C_MUL_ARR_AND_GLOB]]}

// CHECK-DAG: ![[IDX_GROUP_PTR]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_PTR]] = distinct !{![[MD_LOOP_PTR]], ![[MP]], ![[IVDEP_PTR:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_PTR]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_PTR]], i32 5}

// CHECK-DAG: ![[IDX_GROUP_STRUCT_ARR]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_STRUCT_ARR]] = distinct !{![[MD_LOOP_STRUCT_ARR]], ![[MP]], ![[IVDEP_STRUCT_ARR:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_STRUCT_ARR]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_STRUCT_ARR]], i32 5}

// CHECK-DAG: ![[IDX_GROUP_STRUCT_PTR]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_STRUCT_PTR]] = distinct !{![[MD_LOOP_STRUCT_PTR]], ![[MP]], ![[IVDEP_STRUCT_PTR:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_STRUCT_PTR]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_STRUCT_PTR]], i32 5}
