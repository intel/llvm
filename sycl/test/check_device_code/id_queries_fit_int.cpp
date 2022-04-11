// RUN: %clangxx -fsycl-device-only -fsycl-id-queries-fit-in-int -fno-sycl-early-optimizations -S %s -flegacy-pass-manager -o - | FileCheck %s
// RUN: %clangxx -fsycl-device-only -fsycl-id-queries-fit-in-int -fno-sycl-early-optimizations -S %s -fno-legacy-pass-manager -o - | FileCheck %s

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

// CHECK-LABEL: _Z10TestIdDim1N2cl4sycl2idILi1EEE
SYCL_EXTERNAL void TestIdDim1(id<1> TestId) {
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id0Get = TestId.get(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id0 = TestId[0];
}

// CHECK-LABEL: _Z10TestIdDim2N2cl4sycl2idILi2EEE
SYCL_EXTERNAL void TestIdDim2(id<2> TestId) {
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id0Get = TestId.get(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id1Get = TestId.get(1);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id0 = TestId[0];
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id1 = TestId[1];
}

// CHECK-LABEL: _Z10TestIdDim3N2cl4sycl2idILi3EEE
SYCL_EXTERNAL void TestIdDim3(id<3> TestId) {
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id0Get = TestId.get(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id1Get = TestId.get(1);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id2Get = TestId.get(2);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id0 = TestId[0];
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id1 = TestId[1];
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id2 = TestId[2];
}
