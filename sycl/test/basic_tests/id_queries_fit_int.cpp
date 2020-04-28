// RUN: %clangxx -fsycl -Xclang -fsycl-is-host -c -S -emit-llvm -o %t -D__SYCL_ID_QUERIES_FIT_IN_INT__=1 %s
// FileCheck %s --input-file %t

#include <CL/sycl.hpp>

using namespace sycl;

int main() {
  item<1, true> TestItem = detail::Builder::createItem<1, true>({3}, {2}, {1});
  // CHECK: define {{.*}} @_ZNK2cl4sycl4itemILi1ELb1EE6get_idEi
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Id = TestItem.get_id(0);
  // CHECK: define {{.*}} @_ZNK2cl4sycl4itemILi1ELb1EE9get_rangeEi
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int Range = TestItem.get_range(0);
  // CHECK: define {{.*}} @_ZNK2cl4sycl4itemILi1ELb1EE13get_linear_idEv
  // CHECK: call void @llvm.assume(i1 {{.*}})
  int LinearId = TestItem.get_linear_id();

  return 0;
}