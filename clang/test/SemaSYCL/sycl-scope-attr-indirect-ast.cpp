// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -triple spir64-unknown-unknown -ast-dump %s | FileCheck %s

// Check that when parallel_for_work_item is reached from the
// parallel_for_work_group lambda through several levels of calls, every
// function on the call path, and its local variables, get an implicit
// work-group SYCLScopeAttr, while functions that do not lead to a
// parallel_for_work_item are left unmarked.

#include "sycl.hpp"

namespace sycl {
void parallel_for_work_item() {}
} // namespace sycl

namespace direct {
template <typename KernelName, typename KernelType>
__attribute__((sycl_kernel)) void
kernel_parallel_for_work_group(const KernelType &KernelFunc) {
  KernelFunc(sycl::group<1>());
  sycl::parallel_for_work_item();
}
} // namespace direct

void leaf(sycl::group<1> G) { G.parallel_for_work_item([&]() {}); }

void mid(sycl::group<1> G) {
  int MidLocal = 0;
  sycl::private_memory<int> MidPM(G);
  leaf(G);
}

void top(sycl::group<1> G) { mid(G); }

// Reaches parallel_for_work_item through 'leaf', which is already marked by the
// time this path is visited.
void alternate(sycl::group<1> G) { leaf(G); }

void plain(int *P) {
  int PlainLocal = 0;
  *P += PlainLocal;
}

int main() {
  sycl::queue Q;
  int *P = nullptr;
  Q.submit([&](sycl::handler &CGH) {
    sycl::kernel_handler KH;
    CGH.parallel_for_work_group<class Kernel>(
        [=](sycl::group<1> G, sycl::kernel_handler) {
          plain(P);
          top(G);
          alternate(G);
        },
        KH);
  });
  direct::kernel_parallel_for_work_group<class DirectKernel>(
      [](sycl::group<1>) {});
  return 0;
}

// The direct call from the SYCL kernel function is seen by the walk.
// CHECK-LABEL: FunctionDecl {{.*}} used parallel_for_work_item 'void ()'
// CHECK: `-SYCLScopeAttr {{.*}} Implicit WorkItem

// CHECK-LABEL: FunctionDecl {{.*}} used leaf 'void (sycl::group<1>)'
// CHECK: `-SYCLScopeAttr {{.*}} Implicit WorkGroup

// Local variables of intermediate functions get a scope too: work-group scope,
// except private_memory which stays work-item scope.
// CHECK-LABEL: FunctionDecl {{.*}} used mid 'void (sycl::group<1>)'
// CHECK: VarDecl {{.*}} MidLocal 'int'
// CHECK-NOT: VarDecl
// CHECK: SYCLScopeAttr {{.*}} Implicit WorkGroup
// CHECK: VarDecl {{.*}} MidPM 'sycl::private_memory<int>'
// CHECK-NOT: VarDecl
// CHECK: SYCLScopeAttr {{.*}} Implicit WorkItem
// CHECK: `-SYCLScopeAttr {{.*}} Implicit WorkGroup

// CHECK-LABEL: FunctionDecl {{.*}} used top 'void (sycl::group<1>)'
// CHECK: `-SYCLScopeAttr {{.*}} Implicit WorkGroup

// CHECK-LABEL: FunctionDecl {{.*}} used alternate 'void (sycl::group<1>)'
// CHECK: `-SYCLScopeAttr {{.*}} Implicit WorkGroup

// A function that does not reach parallel_for_work_item is not marked, nor are
// its locals.
// CHECK-LABEL: FunctionDecl {{.*}} used plain 'void (int *)'
// CHECK-NOT: SYCLScopeAttr
// CHECK-LABEL: FunctionDecl {{.*}} main 'int ()'

// The generated kernel for DirectKernel calls parallel_for_work_item directly,
// but the walk stops there, so the kernel itself is not marked. Only its kernel
// object gets a (work-group) scope. It is the last decl dumped, and its
// attributes come last, so the NOT runs to the end of the dump, which is where
// an added SYCLScopeAttr would appear.
// CHECK-LABEL: FunctionDecl {{.*}} _ZTSZ4mainE12DirectKernel 'void ()
// CHECK: VarDecl {{.*}} __SYCLKernel
// CHECK: SYCLScopeAttr {{.*}} Implicit WorkGroup
// CHECK: SYCLKernelAttr
// CHECK-NOT: SYCLScopeAttr
