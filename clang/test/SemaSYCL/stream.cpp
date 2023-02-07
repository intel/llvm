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

// Function Declaration
// CHECK: FunctionDecl {{.*}}stream_test{{.*}}

// Initializers:

// 'in_lambda'
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .in_lambda
// CHECK-NEXT:      DeclRefExpr
// in_lambda __init
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at

// 'in_lambda_array'
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .in_lambda_array
// CHECK-NEXT:        DeclRefExpr
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .in_lambda_array
// CHECK-NEXT:        DeclRefExpr
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr  {{.*}} 'sycl::stream[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// 'in_lambda_mdarray'
// [0][0]
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       ArraySubscriptExpr
// CHECK-NEXT:        ImplicitCastExpr
// CHECK-NEXT:         MemberExpr {{.*}} .in_lambda_mdarray
// CHECK-NEXT:          DeclRefExpr
// CHECK-NEXT:       IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// [1][0]
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       ArraySubscriptExpr
// CHECK-NEXT:        ImplicitCastExpr
// CHECK-NEXT:         MemberExpr {{.*}} .in_lambda_mdarray
// CHECK-NEXT:          DeclRefExpr
// CHECK-NEXT:       IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// [0][1]
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       ArraySubscriptExpr
// CHECK-NEXT:        ImplicitCastExpr
// CHECK-NEXT:         MemberExpr {{.*}} .in_lambda_mdarray
// CHECK-NEXT:          DeclRefExpr
// CHECK-NEXT:       IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// [1][1]
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       ArraySubscriptExpr
// CHECK-NEXT:        ImplicitCastExpr
// CHECK-NEXT:         MemberExpr {{.*}} .in_lambda_mdarray
// CHECK-NEXT:          DeclRefExpr
// CHECK-NEXT:       IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// HasStreams struct
// HasStreams::s1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .s1
// CHECK-NEXT:      MemberExpr {{.*}} .Struct
// CHECK-NEXT:       DeclRefExpr
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: MemberExpr {{.*}}'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// 'HasStreams::s_array'
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        MemberExpr {{.*}} .Struct
// CHECK-NEXT:         DeclRefExpr
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}}'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        MemberExpr {{.*}} .Struct
// CHECK-NEXT:         DeclRefExpr
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}}'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// HasArrayOfHasStreams
// HasArrayOfHasStreams::i
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .i
// CHECK-NEXT:     MemberExpr {{.*}} .haohs
// CHECK-NEXT:      DeclRefExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_i'
// CHECK-NEXT:  IntegerLiteral {{.*}} 4
// HasArrayOfHasStreams::hs
// HasStreams struct
// HasStreams::s1
// CHECK-NEXT: CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .s1
// CHECK-NEXT:      ArraySubscriptExpr
// CHECK-NEXT:       ImplicitCastExpr
// CHECK-NEXT:        MemberExpr {{.*}} .hs
// CHECK-NEXT:         MemberExpr {{.*}} .haohs
// CHECK-NEXT:          DeclRefExpr
// CHECK-NEXT:      IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// HasStreams::s_array
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           MemberExpr {{.*}} .haohs
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:        IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           MemberExpr {{.*}} .haohs
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:        IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// HasStreams struct
// HasStreams::s1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .s1
// CHECK-NEXT:      ArraySubscriptExpr
// CHECK-NEXT:       ImplicitCastExpr
// CHECK-NEXT:        MemberExpr {{.*}} .hs
// CHECK-NEXT:         MemberExpr {{.*}} .haohs
// CHECK-NEXT:          DeclRefExpr
// CHECK-NEXT:      IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// HasStreams::s_array
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           MemberExpr {{.*}} .haohs
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:        IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           MemberExpr {{.*}} .haohs
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:        IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// HasArrayOfHasStreams Array
// HasArrayOfHasStreams Struct
// HasArrayOfHasStreams::i
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .i
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .haohs
// CHECK-NEXT:        DeclRefExpr
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_i'
// CHECK-NEXT:  IntegerLiteral {{.*}} 4
// HasArrayOfHasStreams::hs
// HasStreams struct
// HasStreams::s1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .s1
// CHECK-NEXT:      ArraySubscriptExpr
// CHECK-NEXT:       ImplicitCastExpr
// CHECK-NEXT:        MemberExpr {{.*}} .hs
// CHECK-NEXT:         ArraySubscriptExpr
// CHECK-NEXT:          ImplicitCastExpr
// CHECK-NEXT:           MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:         IntegerLiteral {{.*}} 0
// CHECK-NEXT:      IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// HasStreams::s_array
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 0
// CHECK-NEXT:        IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 0
// CHECK-NEXT:        IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// HasStreams struct
// HasStreams::s1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .s1
// CHECK-NEXT:      ArraySubscriptExpr
// CHECK-NEXT:       ImplicitCastExpr
// CHECK-NEXT:        MemberExpr {{.*}} .hs
// CHECK-NEXT:         ArraySubscriptExpr
// CHECK-NEXT:          ImplicitCastExpr
// CHECK-NEXT:           MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:         IntegerLiteral {{.*}} 0
// CHECK-NEXT:      IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// HasStreams::s_array
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 0
// CHECK-NEXT:        IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 0
// CHECK-NEXT:        IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}  'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// HasArrayOfHasStreams Struct
// HasArrayOfHasStreams::i
// CHECK:      CallExpr
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   DeclRefExpr {{.*}} '__builtin_memcpy'
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    MemberExpr {{.*}} .i
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .haohs
// CHECK-NEXT:        DeclRefExpr
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT:  ImplicitCastExpr
// CHECK-NEXT:   UnaryOperator
// CHECK-NEXT:    DeclRefExpr {{.*}} '_arg_i'
// CHECK-NEXT:  IntegerLiteral {{.*}} 4
// HasArrayOfHasStreams::hs
// HasStreams struct
// HasStreams::s1
// CHECK-NEXT: CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .s1
// CHECK-NEXT:      ArraySubscriptExpr
// CHECK-NEXT:       ImplicitCastExpr
// CHECK-NEXT:        MemberExpr {{.*}} .hs
// CHECK-NEXT:         ArraySubscriptExpr
// CHECK-NEXT:          ImplicitCastExpr
// CHECK-NEXT:           MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:         IntegerLiteral {{.*}} 1
// CHECK-NEXT:      IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}  'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// HasStreams::s_array
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 1
// CHECK-NEXT:        IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}  'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 1
// CHECK-NEXT:        IntegerLiteral {{.*}} 0
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}  'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// HasStreams struct
// HasStreams::s1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     MemberExpr {{.*}} .s1
// CHECK-NEXT:      ArraySubscriptExpr
// CHECK-NEXT:       ImplicitCastExpr
// CHECK-NEXT:        MemberExpr {{.*}} .hs
// CHECK-NEXT:         ArraySubscriptExpr
// CHECK-NEXT:          ImplicitCastExpr
// CHECK-NEXT:           MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:            DeclRefExpr
// CHECK-NEXT:         IntegerLiteral {{.*}} 1
// CHECK-NEXT:      IntegerLiteral {{.*}} 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}  'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// HasStreams::s_array
// element 0
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 1
// CHECK-NEXT:        IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 0
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}  'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK:      CXXNewExpr
// CHECK-NEXT:  CXXConstructExpr
// CHECK-NEXT:   ImplicitCastExpr
// CHECK-NEXT:    UnaryOperator
// CHECK-NEXT:     ArraySubscriptExpr
// CHECK-NEXT:      ImplicitCastExpr
// CHECK-NEXT:       MemberExpr {{.*}} .s_array
// CHECK-NEXT:        ArraySubscriptExpr
// CHECK-NEXT:         ImplicitCastExpr
// CHECK-NEXT:          MemberExpr {{.*}} .hs
// CHECK-NEXT:           ArraySubscriptExpr
// CHECK-NEXT:            ImplicitCastExpr
// CHECK-NEXT:             MemberExpr {{.*}} .haohs_array
// CHECK-NEXT:              DeclRefExpr
// CHECK-NEXT:           IntegerLiteral {{.*}} 1
// CHECK-NEXT:        IntegerLiteral {{.*}} 1
// CHECK-NEXT:     IntegerLiteral {{.*}} 1
// CHECK-NEXT: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}}  'void (__global char *, range<1>, range<1>, id<1>, int)' lvalue .__init
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1


// Finalize
// in_lambda __finalize
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at

// _in_lambda_array
// element 0
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// _in_lambda_mdarray
// [0][0]
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// [0][1]
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// [1][0]
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// [1][1]
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream[2]' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream (*)[2]' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2][2]' lvalue
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// HasStreams
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: MemberExpr {{.*}}'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// array:
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}}'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: MemberExpr {{.*}}'HasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// HasArrayOfHasStreams
// First element
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// array:
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// second element
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// array:
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: MemberExpr {{.*}}'HasArrayOfHasStreams' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1

// HasArrayOfHasStreams array
// First element
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// array:
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// second element
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// array:
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// second element
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// array:
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// second element
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue .s1
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// array:
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 0
// element 1
// CHECK: CXXMemberCallExpr {{.*}} 'void'
// CHECK-NEXT: MemberExpr {{.*}} 'void ()' lvalue .__finalize
// CHECK-NEXT: ArraySubscriptExpr {{.*}} 'sycl::stream':'sycl::stream' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'sycl::stream *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'sycl::stream[2]' lvalue .s_array
// CHECK-NEXT: ArraySubscriptExpr {{.*}}'HasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasStreams[2]' lvalue .hs
// CHECK-NEXT: ArraySubscriptExpr{{.*}}'HasArrayOfHasStreams' lvalue
// CHECK-NEXT: ImplicitCastExpr {{.*}} 'HasArrayOfHasStreams *' <ArrayToPointerDecay>
// CHECK-NEXT: MemberExpr {{.*}} 'HasArrayOfHasStreams[2]' lvalue .
// CHECK-NEXT: DeclRefExpr {{.*}} '(lambda at
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
// CHECK-NEXT: IntegerLiteral {{.*}} '{{.*}}' 1
