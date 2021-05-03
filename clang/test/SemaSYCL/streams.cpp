// RUN: %clang_cc1 -S -fsycl-is-device -internal-isystem %S/Inputs -triple spir64 -ast-dump -sycl-std=2020 %s | FileCheck %s

// This test demonstrates passing of SYCL stream instances as kernel arguments and checks if the compiler generates
// the correct ast-dump

#include "sycl.hpp"

sycl::queue myQueue;

sycl::handler H;

struct HasStreams {
  sycl::stream s1{0, 0, H}; // stream(totalBufferSize, workItemBufferSize, handler)

  sycl::stream s_array[2] = {{0, 0, H}, {0, 0, H}};
};

struct HasArrayOfHasStreams {
  int i;
  HasStreams hs[2];
};

int main() {
  sycl::stream in_lambda{0, 0, H};
  sycl::stream in_lambda_array[2] = {{0, 0, H}, {0, 0, H}};
  sycl::stream in_lambda_mdarray[2][2] = {{{0, 0, H}, {0, 0, H}}, {{0, 0, H}, {0, 0, H}}};

  HasStreams Struct;
  HasArrayOfHasStreams haohs;
  HasArrayOfHasStreams haohs_array[2];

  myQueue.submit([&](sycl::handler &h) {
    h.single_task<class stream_test>([=]() {
      in_lambda.use();
      in_lambda_array[1].use();
      in_lambda_mdarray[1][1].use();

      Struct.s1.use();

      haohs.hs[0].s1.use();
      haohs_array[0].hs[0].s1.use();
    });
  });

  return 0;
}
// CHECK: FunctionDecl {{.*}} main 'int ()'
// CHECK: FunctionDecl {{.*}}stream_test
// CHECK: InitListExpr {{.*}} '(lambda at

// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2][2]'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'


// HasStreams struct
// CHECK: InitListExpr {{.*}} 'HasStreams'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'HasArrayOfHasStreams'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar
// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams [2]'
// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'

// HasArrayOfHasStreams struct
// CHECK: InitListExpr {{.*}} 'HasArrayOfHasStreams [2]'
// CHECK-NEXT: InitListExpr {{.*}} 'HasArrayOfHasStreams'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar

// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams [2]'
// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'

// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'

// CHECK-NEXT: InitListExpr {{.*}} 'HasArrayOfHasStreams'
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'int' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} 'int' lvalue ParmVar

// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams [2]'
// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'

// CHECK-NEXT: InitListExpr {{.*}} 'HasStreams'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: InitListExpr {{.*}} 'sycl::stream [2]'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'
// CHECK-NEXT: CXXConstructExpr {{.*}} 'sycl::stream':'sycl::stream' 'void () noexcept'

// Calls to init
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at

//_in_lambda_array
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0


// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0


// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK:  CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream [2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2][2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: ImplicitCastExpr {{.*}} '__global char *' <LValueToRValue>
// CHECK-NEXT: DeclRefExpr {{.*}} '__global char *' lvalue ParmVar {{.*}} '_arg_s1' '__global char *'

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK:  | |-CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0


// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0


// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream [2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams [2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams [2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
