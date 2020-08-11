// RUN: %clang_cc1 -I %S/Inputs -fsycl -fsycl-is-device -ast-dump %s | FileCheck %s
// expected-no-diagnostics

// This test checks that compiler generates correct kernel arguments for
// union without array.

#include <sycl.hpp>

using namespace cl::sycl;

typedef float realw;

typedef union dpct_type_54e08f {
  float cuda;
} gpu_realw_mem;

void call_some_dummy_kernel(float data) {
  data = 2.0f;
}

template <typename Name, typename Type>
__attribute__((sycl_kernel)) void parallel_for(Type lambda) {
  lambda();
}

int main() {
  gpu_realw_mem accel;

  parallel_for<class kernel>(
      [=]() {
        call_some_dummy_kernel(accel.cuda);
      });
}

// Check kernel parameters
// CHECK: FunctionDecl {{.*}}kernel{{.*}} 'void (gpu_realw_mem, float)'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_ 'gpu_realw_mem':'dpct_type_54e08f'
// CHECK-NEXT: ParmVarDecl {{.*}} used _arg_cuda 'float'

// Check kernel inits
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: DeclStmt
// CHECK-NEXT: VarDecl {{.*}} cinit
// CHECK-NEXT: InitListExpr
// CHECK-NEXT: CXXConstructExpr {{.*}} 'gpu_realw_mem':'dpct_type_54e08f' 'void (const dpct_type_54e08f &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const dpct_type_54e08f'
// CHECK-NEXT: DeclRefExpr {{.*}} 'gpu_realw_mem':'dpct_type_54e08f' lvalue ParmVar {{.*}} '_arg_' 'gpu_realw_mem':'dpct_type_54e08f'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'float'
// CHECK-NEXT: DeclRefExpr {{.*}} 'float' lvalue ParmVar {{.*}} '_arg_cuda' 'float'
