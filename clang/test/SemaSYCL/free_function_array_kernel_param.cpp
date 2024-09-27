// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-device -ast-dump \
// RUN: %s -o - | FileCheck %s

// This test checks parameter rewriting for free functions with parameters
// of type struct with array and array of pointers.

#include "sycl.hpp"

constexpr int TestArrSize = 3;

template <int ArrSize>
struct KArgWithPtrArray {
  int *data[ArrSize];
  int start[ArrSize];
  int end[ArrSize];
  constexpr int getArrSize() { return ArrSize; }
};

template <int ArrSize>
[[__sycl_detail__::add_ir_attributes_function("sycl-single-task-kernel", 0)]]
void ff_6(KArgWithPtrArray<ArrSize> KArg) {
  for (int j = 0; j < ArrSize; j++)
    for (int i = KArg.start[j]; i <= KArg.end[j]; i++)
      KArg.data[j][i] = KArg.start[j] + KArg.end[j];
}

template void ff_6(KArgWithPtrArray<TestArrSize> KArg);

// CHECK: FunctionDecl {{.*}}__sycl_kernel{{.*}}'void (__generated_KArgWithPtrArray)'
// CHECK-NEXT: ParmVarDecl {{.*}} used __arg_KArg '__generated_KArgWithPtrArray'
// CHECK-NEXT: CompoundStmt
// CHECK-NEXT: CallExpr
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'void (*)(KArgWithPtrArray<3>)' <FunctionToPointerDecay>
// CHECK-NEXT: DeclRefExpr {{.*}} 'void (KArgWithPtrArray<3>)' lvalue Function {{.*}} 'ff_6' 'void (KArgWithPtrArray<3>)'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'KArgWithPtrArray<3>' 'void (const KArgWithPtrArray<3> &) noexcept'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'const KArgWithPtrArray<3>' lvalue <NoOp>
// CHECK-NEXT: UnaryOperator {{.*}} 'KArgWithPtrArray<3>' lvalue prefix '*' cannot overflow
// CHECK-NEXT: CXXReinterpretCastExpr {{.*}} 'KArgWithPtrArray<3> *' reinterpret_cast<KArgWithPtrArray<3> *> <BitCast>
// CHECK-NEXT: UnaryOperator {{.*}} '__generated_KArgWithPtrArray *' prefix '&' cannot overflow
// CHECK-NEXT: DeclRefExpr {{.*}} '__generated_KArgWithPtrArray' lvalue ParmVar {{.*}} '__arg_KArg' '__generated_KArgWithPtrArray'
