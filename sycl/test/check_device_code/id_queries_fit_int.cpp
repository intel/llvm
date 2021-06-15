// RUN: %clangxx -fsycl-device-only -fsycl-id-queries-fit-in-int -fno-sycl-early-optimizations -S -o %t.ll %s
// RUN: FileCheck %s --input-file %t.ll

#include <CL/sycl.hpp>

using namespace sycl;

// CHECK: define {{.*}}dso_local spir_func void @{{.*}}testItem{{.*}}(%"class.{{.*}}item"*{{.*}}%{{.*}})
SYCL_EXTERNAL void testItem(item<1> TestItem) {
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id = TestItem.get_id(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Range = TestItem.get_range(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int LinearId = TestItem.get_linear_id();
}

// CHECK: define {{.*}}dso_local spir_func void @{{.*}}testNDItem{{.*}}(%"class.{{.*}}nd_item"*{{.*}}%{{.*}})
SYCL_EXTERNAL void testNDItem(nd_item<1> TestNDItem) {
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int GlobalId = TestNDItem.get_global_id(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int GlobalLinearId = TestNDItem.get_global_linear_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int LocalId = TestNDItem.get_local_id(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int LocalLinearId = TestNDItem.get_local_linear_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int GroupRange = TestNDItem.get_group_range(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int GroupId = TestNDItem.get_group(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int GroupLinearId = TestNDItem.get_group_linear_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int GlobalRange = TestNDItem.get_global_range(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int LocalRange = TestNDItem.get_local_range(0);

  // CHECK: call void @llvm.assume(i1 {{.*}})
  int GlobalIdConverted = TestNDItem.get_global_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int LocalIdConverted = TestNDItem.get_local_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int OffsetConferted = TestNDItem.get_offset();
}
