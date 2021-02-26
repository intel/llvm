// RUN: %clang_cc1 -triple spir64-unknown-unknown-sycldevice -disable-llvm-passes -fsycl-is-device -emit-llvm %s -o - | FileCheck %s

// Accesses from the inner loop only, various global safelens for the outer and the inner loops.
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_inner_loop_accessv()
void ivdep_inner_loop_access() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  [[intel::ivdep]] for (int i = 0; i != 10; ++i) {
    [[intel::ivdep(3)]] for (int j = 0; j != 10; ++j) {
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_INNER_ACCESS:[0-9]+]]
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_INNER_ACCESS]]
      a[i] = a[(i + j) % 10];
      // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_INNER_LOOP_INNER_ACCESS:[0-9]+]]
    }
    // CHECK: br label %for.cond, !llvm.loop ![[MD_OUTER_LOOP_INNER_ACCESS:[0-9]+]]
  }
}

// Accesses from both inner and outer loop, same global (INF) safelen for both.
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_embedded_global_safelenv()
void ivdep_embedded_global_safelen() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  [[intel::ivdep]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_OUTER_GLOB_SFLN:[0-9]+]]
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_OUTER_GLOB_SFLN]]
    a[i] = a[i % 2];
    [[intel::ivdep]] for (int j = 0; j != 10; ++j) {
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_INNER_GLOB_SFLN:[0-9]+]]
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_INNER_GLOB_SFLN]]
      a[i] = a[(i + j) % 10];
      // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_INNER_GLOB_SFLN:[0-9]+]]
    }
    // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_OUTER_GLOB_SFLN:[0-9]+]]
  }
}

// Accesses from both inner and outer loop, with various safelens per loop.
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_embedded_various_safelensv()
void ivdep_embedded_various_safelens() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  [[intel::ivdep(a, 4)]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_OUTER_VAR_SFLN:[0-9]+]]
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_OUTER_VAR_SFLN]]
    a[i] = a[i % 2];
    [[intel::ivdep(a, 2)]] for (int j = 0; j != 10; ++j) {
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_INNER_VAR_SFLN:[0-9]+]]
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_INNER_VAR_SFLN]]
      a[i] = a[(i + j) % 10];
      // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_INNER_VAR_SFLN:[0-9]+]]
    }
    // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_OUTER_VAR_SFLN:[0-9]+]]
  }
}

// Multiple arrays accessed from both loops.
// Outer loop: array-specific ivdeps for all arrays with various safelens
// Inner loop: global ivdep with its own safelen
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_embedded_multiple_arraysv()
void ivdep_embedded_multiple_arrays() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep(a, 3), intel::ivdep(b, 4)]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_OUTER_MUL_ARRS:[0-9]+]]
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_OUTER_MUL_ARRS]]
    a[i] = a[i % 2];
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_OUTER_MUL_ARRS:[0-9]+]]
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_OUTER_MUL_ARRS]]
    b[i] = b[i % 2];
    [[intel::ivdep(2)]] for (int j = 0; j != 10; ++j) {
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_INNER_MUL_ARRS:[0-9]+]]
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_INNER_MUL_ARRS:[0-9]+]]
      a[i] = b[(i + j) % 10];
      // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_INNER_MUL_ARRS:[0-9]+]]
    }
    // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_OUTER_MUL_ARRS:[0-9]+]]
  }
}

// Multiple arrays accessed from both loops.
// Outer loop: array-specific ivdep for one of the arrays
// Inner loop: global ivdep (i.e. applies to all arrays)
// As the outer loop's ivdep applies to a particular, other array(s) shouldn't be marked
// into any index group at the outer loop level
//
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_embedded_multiple_arrays_globalv()
void ivdep_embedded_multiple_arrays_global() {
  // CHECK: %[[ARRAY_A:[0-9a-z]+]] = alloca [10 x i32]
  int a[10];
  // CHECK: %[[ARRAY_B:[0-9a-z]+]] = alloca [10 x i32]
  int b[10];
  [[intel::ivdep(a)]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_OUTER_MUL_ARRS_GLOB:[0-9]+]]
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_OUTER_MUL_ARRS_GLOB]]
    a[i] = a[i % 2];
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}{{[[:space:]]}}
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}{{[[:space:]]}}
    b[i] = b[i % 2];
    [[intel::ivdep]] for (int j = 0; j != 10; ++j) {
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_B]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_B_INNER_MUL_ARRS_GLOB:[0-9]+]]
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_A_INNER_MUL_ARRS_GLOB:[0-9]+]]
      a[i] = b[(i + j) % 10];
      // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_INNER_MUL_ARRS_GLOB:[0-9]+]]
    }
    // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_OUTER_MUL_ARRS_GLOB:[0-9]+]]
  }
}

// Accesses within each dimension of a multi-dimensional (n > 2) loop
// CHECK: define {{.*}}spir_func void @_Z{{[0-9]+}}ivdep_embedded_multiple_dimensionsv()
void ivdep_embedded_multiple_dimensions() {
  int a[10];
  [[intel::ivdep]] for (int i = 0; i != 10; ++i) {
    // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_DIM_1_MUL_DIMS:[0-9]+]]
    a[i] = i;
    [[intel::ivdep]] for (int j = 0; j != 10; ++j) {
      // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_DIM_2_MUL_DIMS:[0-9]+]]
      a[j] += j;
      [[intel::ivdep]] for (int k = 0; k != 10; ++k) {
        // CHECK: %{{[0-9a-z]+}} = getelementptr inbounds [10 x i32], [10 x i32] addrspace(4)* %[[ARRAY_A]].ascast, i64 0, i64 %{{[0-9a-z]+}}, !llvm.index.group ![[IDX_GROUP_DIM_3_MUL_DIMS:[0-9]+]]
        a[k] += k;
        // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_DIM_3_MUL_DIMS:[0-9]+]]
      }
      // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_DIM_2_MUL_DIMS:[0-9]+]]
    }
    // CHECK: br label %for.cond{{[0-9]*}}, !llvm.loop ![[MD_LOOP_DIM_1_MUL_DIMS:[0-9]+]]
  }
}

template <typename name, typename Func>
__attribute__((sycl_kernel)) void kernel_single_task(const Func &kernelFunc) {
  kernelFunc();
}

int main() {
  kernel_single_task<class kernel_function>([]() {
    ivdep_inner_loop_access();
    ivdep_embedded_global_safelen();
    ivdep_embedded_various_safelens();
    ivdep_embedded_multiple_arrays();
    ivdep_embedded_multiple_arrays_global();
    ivdep_embedded_multiple_dimensions();
  });
  return 0;
}

// Find the single instance of a legacy "IVDep enable" MD node.
// CHECK-DAG: ![[IVDEP_LEGACY_ENABLE:[0-9]+]] = !{!"llvm.loop.ivdep.enable"}

/// Accesses from the inner loop only, various global safelens for the outer and the inner loops.
/// The inner loop's index group(s) should have two subnodes (outer-loop node and inner-loop node).
//
// Inner loop
// CHECK-DAG: ![[IDX_GROUP_INNER_ACCESS]] = !{![[OUTER_NODE_INNER_ACCESS:[0-9]+]], ![[INNER_NODE_INNER_ACCESS:[0-9]+]]}
// CHECK-DAG: ![[INNER_NODE_INNER_ACCESS]] = distinct !{}
// CHECK-DAG: ![[MD_INNER_LOOP_INNER_ACCESS]] = distinct !{![[MD_INNER_LOOP_INNER_ACCESS]], ![[#]], ![[IVDEP_INNER_INNER_ACCESS:[0-9]+]], ![[IVDEP_LEGACY_INNER_INNER_ACCESS:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_LEGACY_INNER_INNER_ACCESS]] = !{!"llvm.loop.ivdep.safelen", i32 3}
// CHECK-DAG: ![[IVDEP_INNER_INNER_ACCESS]] = !{!"llvm.loop.parallel_access_indices", ![[INNER_NODE_INNER_ACCESS]], i32 3}
//
// Outer loop
// CHECK-DAG: ![[OUTER_NODE_INNER_ACCESS]] = distinct !{}
// CHECK-DAG: ![[MD_OUTER_LOOP_INNER_ACCESS]] = distinct !{![[MD_OUTER_LOOP_INNER_ACCESS]], ![[#]], ![[IVDEP_OUTER_INNER_ACCESS:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_OUTER_INNER_ACCESS]] = !{!"llvm.loop.parallel_access_indices", ![[OUTER_NODE_INNER_ACCESS]]}

/// Accesses from both inner and outer loop, same global (INF) safelen for both.
/// The "outer loop subnode" of the inner loop's index group points at the outer loop's index group
//
// Inner loop
// CHECK-DAG: ![[IDX_GROUP_INNER_GLOB_SFLN]] = !{![[IDX_GROUP_OUTER_GLOB_SFLN]], ![[INNER_NODE_GLOB_SFLN:[0-9]+]]}
// CHECK-DAG: ![[INNER_NODE_GLOB_SFLN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_INNER_GLOB_SFLN]] = distinct !{![[MD_LOOP_INNER_GLOB_SFLN]], ![[#]], ![[IVDEP_INNER_GLOB_SFLN:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_INNER_GLOB_SFLN]] = !{!"llvm.loop.parallel_access_indices", ![[INNER_NODE_GLOB_SFLN]]}
//
// Outer loop
// CHECK-DAG: ![[IDX_GROUP_OUTER_GLOB_SFLN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_OUTER_GLOB_SFLN]] = distinct !{![[MD_LOOP_OUTER_GLOB_SFLN]], ![[#]], ![[IVDEP_OUTER_GLOB_SFLN:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_OUTER_GLOB_SFLN]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_OUTER_GLOB_SFLN]]}

/// Accesses from both inner and outer loop, with various safelens per loop.
/// The "outer loop subnode" of the inner loop's index group points at the outer loop's index group
//
// Inner loop
// CHECK-DAG: ![[IDX_GROUP_INNER_VAR_SFLN]] = !{![[IDX_GROUP_OUTER_VAR_SFLN]], ![[INNER_NODE_VAR_SFLN:[0-9]+]]}
// CHECK-DAG: ![[INNER_NODE_VAR_SFLN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_INNER_VAR_SFLN]] = distinct !{![[MD_LOOP_INNER_VAR_SFLN]], ![[#]], ![[IVDEP_INNER_VAR_SFLN:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_INNER_VAR_SFLN]] = !{!"llvm.loop.parallel_access_indices", ![[INNER_NODE_VAR_SFLN]], i32 2}
//
// Outer loop
// CHECK-DAG: ![[IDX_GROUP_OUTER_VAR_SFLN]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_OUTER_VAR_SFLN]] = distinct !{![[MD_LOOP_OUTER_VAR_SFLN]], ![[#]], ![[IVDEP_OUTER_VAR_SFLN:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_OUTER_VAR_SFLN]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_OUTER_VAR_SFLN]], i32 4}

/// Multiple arrays accessed from both loops.
/// The "outer loop subnode" of the inner loop's index group points at the outer loop's index group
//
// Inner loop
// CHECK-DAG: ![[IDX_GROUP_A_INNER_MUL_ARRS]] = !{![[IDX_GROUP_A_OUTER_MUL_ARRS]], ![[INNER_NODE_A_MUL_ARRS:[0-9]+]]}
// CHECK-DAG: ![[INNER_NODE_A_MUL_ARRS]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_INNER_MUL_ARRS]] = !{![[IDX_GROUP_B_OUTER_MUL_ARRS]], ![[INNER_NODE_B_MUL_ARRS:[0-9]+]]}
// CHECK-DAG: ![[INNER_NODE_B_MUL_ARRS]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_INNER_MUL_ARRS]] = distinct !{![[MD_LOOP_INNER_MUL_ARRS]], ![[#]], ![[IVDEP_INNER_A_B_MUL_ARRS:[0-9]+]], ![[IVDEP_LEGACY_INNER_MUL_ARRS:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_LEGACY_INNER_MUL_ARRS]] = !{!"llvm.loop.ivdep.safelen", i32 2}
// CHECK-DAG: ![[IVDEP_INNER_A_B_MUL_ARRS]] = !{!"llvm.loop.parallel_access_indices", ![[INNER_NODE_B_MUL_ARRS]], ![[INNER_NODE_A_MUL_ARRS]], i32 2}
//
// Outer loop
// CHECK-DAG: ![[IDX_GROUP_A_OUTER_MUL_ARRS]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_OUTER_MUL_ARRS]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_OUTER_MUL_ARRS]] = distinct !{![[MD_LOOP_OUTER_MUL_ARRS]], ![[#]], ![[IVDEP_OUTER_A_MUL_ARRS:[0-9]+]], ![[IVDEP_OUTER_B_MUL_ARRS:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_OUTER_A_MUL_ARRS]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_OUTER_MUL_ARRS]], i32 3}
// CHECK-DAG: ![[IVDEP_OUTER_B_MUL_ARRS]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_B_OUTER_MUL_ARRS]], i32 4}

/// Multiple arrays accessed from both loops.
/// The "outer loop subnode" of the inner loop's index group points at the outer loop's index group
//
// Inner loop
// CHECK-DAG: ![[IDX_GROUP_A_INNER_MUL_ARRS_GLOB]] = !{![[IDX_GROUP_A_OUTER_MUL_ARRS_GLOB]], ![[INNER_NODE_A_MUL_ARRS_GLOB:[0-9]+]]}
// CHECK-DAG: ![[INNER_NODE_A_MUL_ARRS_GLOB]] = distinct !{}
// CHECK-DAG: ![[IDX_GROUP_B_INNER_MUL_ARRS_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_INNER_MUL_ARRS_GLOB]] = distinct !{![[MD_LOOP_INNER_MUL_ARRS_GLOB]], ![[#]], ![[IVDEP_INNER_A_B_MUL_ARRS_GLOB:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_INNER_A_B_MUL_ARRS_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_B_INNER_MUL_ARRS_GLOB]], ![[INNER_NODE_A_MUL_ARRS_GLOB]]}
//
// Outer loop
// CHECK-DAG: ![[IDX_GROUP_A_OUTER_MUL_ARRS_GLOB]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_OUTER_MUL_ARRS_GLOB]] = distinct !{![[MD_LOOP_OUTER_MUL_ARRS_GLOB]], ![[#]], ![[IVDEP_OUTER_A_MUL_ARRS_GLOB:[0-9]+]]}
// CHECK-DAG: ![[IVDEP_OUTER_A_MUL_ARRS_GLOB]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_A_OUTER_MUL_ARRS_GLOB]]}

/// Accesses within each dimension of a multi-dimensional (n > 2) loop
/// Index group(s) of each inner loop should have a subnode that points to the containing loop's index group subnode
/// (in case the containing loop is the outermost, to the index group itself)
//
// Loop dimension 3 (the innermost loop)
// CHECK-DAG: ![[IDX_GROUP_DIM_3_MUL_DIMS]] = !{![[IDX_GROUP_DIM_1_MUL_DIMS]], ![[DIM_2_NODE_MUL_DIMS:[0-9]+]], ![[DIM_3_NODE_MUL_DIMS:[0-9]+]]}
// CHECK-DAG: ![[DIM_3_NODE_MUL_DIMS]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_DIM_3_MUL_DIMS]] = distinct !{![[MD_LOOP_DIM_3_MUL_DIMS]], ![[#]], ![[IVDEP_DIM_3_MUL_DIMS:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_DIM_3_MUL_DIMS]] = !{!"llvm.loop.parallel_access_indices", ![[DIM_3_NODE_MUL_DIMS]]}
//
// Loop dimension 2
// CHECK-DAG: ![[IDX_GROUP_DIM_2_MUL_DIMS]] = !{![[IDX_GROUP_DIM_1_MUL_DIMS]], ![[DIM_2_NODE_MUL_DIMS]]}
// CHECK-DAG: ![[DIM_2_NODE_MUL_DIMS]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_DIM_2_MUL_DIMS]] = distinct !{![[MD_LOOP_DIM_2_MUL_DIMS]], ![[#]], ![[IVDEP_DIM_2_MUL_DIMS:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_DIM_2_MUL_DIMS]] = !{!"llvm.loop.parallel_access_indices", ![[DIM_2_NODE_MUL_DIMS]]}
//
// Loop dimension 1 (the outermost loop)
// CHECK-DAG: ![[IDX_GROUP_DIM_1_MUL_DIMS]] = distinct !{}
// CHECK-DAG: ![[MD_LOOP_DIM_1_MUL_DIMS]] = distinct !{![[MD_LOOP_DIM_1_MUL_DIMS]], ![[#]], ![[IVDEP_DIM_1_MUL_DIMS:[0-9]+]], ![[IVDEP_LEGACY_ENABLE]]}
// CHECK-DAG: ![[IVDEP_DIM_1_MUL_DIMS]] = !{!"llvm.loop.parallel_access_indices", ![[IDX_GROUP_DIM_1_MUL_DIMS]]}
