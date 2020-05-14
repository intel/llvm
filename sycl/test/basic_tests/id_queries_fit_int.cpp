// RUN: %clangxx -fsycl -Xclang -fsycl-is-host -O1 -c -S -emit-llvm -o %t.ll -D__SYCL_ID_QUERIES_FIT_IN_INT__=1 %s
// RUN: FileCheck %s --input-file %t.ll

#include <CL/sycl.hpp>

using namespace sycl;

// CHECK: define dso_local i32 @main() {{.*}} {
int main() {
  item<1, true> TestItem = detail::Builder::createItem<1, true>({3}, {2}, {1});
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id = TestItem.get_id(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Range = TestItem.get_range(0);
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int LinearId = TestItem.get_linear_id();

  id<1> IdD = TestItem.get_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  range<1> RangeD = TestItem.get_range();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  id<1> Offset = TestItem.get_offset();
  // CHECK: call void @llvm.assume(i1 {{.*}})

  cl::sycl::nd_item<1> TestNDItem =
      detail::Builder::createNDItem<1>(detail::Builder::createItem<1, false>({4}, {2}),
                                       detail::Builder::createItem<1, false>({2}, {0}),
                                       detail::Builder::createGroup<1>({4}, {2}, {1}));

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

  id<1> GlobalIdD = TestNDItem.get_global_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  id<1> LocalIdD = TestNDItem.get_local_id();
  // CHECK: call void @llvm.assume(i1 {{.*}})
  group<1> GroupD = TestNDItem.get_group();

  return 0;
}
// CHECK: }
