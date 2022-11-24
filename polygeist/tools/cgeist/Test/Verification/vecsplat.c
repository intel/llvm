// RUN: cgeist %s --function=* -S | FileCheck %s
// RUN: cgeist %s --function=* -S -emit-llvm | FileCheck %s --check-prefix=CHECK-LLVM

typedef int int_t_vec __attribute__((ext_vector_type(3)));
typedef float float_t_vec __attribute__((ext_vector_type(4)));

// CHECK-LABEL:   func.func @splat_int(
// CHECK-SAME:                         %[[VAL_0:.*]]: i32) -> vector<3xi32>
// CHECK-NEXT:      %[[VAL_1:.*]] = vector.splat %[[VAL_0]] : vector<3xi32>
// CHECK-NEXT:      return %[[VAL_1]] : vector<3xi32>
// CHECK-NEXT:    }

// CHECK-LLVM-LABEL:    define <3 x i32> @splat_int(
// CHECK-LLVM-SAME:                                 i32 %[[VAL_0:.*]]) {
// CHECK-LLVM-NEXT:       %[[VAL_1:.*]] = insertelement <3 x i32> undef, i32 %[[VAL_0]], i32 0
// CHECK-LLVM-NEXT:       %[[VAL_2:.*]] = shufflevector <3 x i32> %[[VAL_1]], <3 x i32> undef, <3 x i32> zeroinitializer
// CHECK-LLVM-NEXT:       ret <3 x i32> %[[VAL_2]]
// CHECK-LLVM-NEXT:     }

int_t_vec splat_int(int el) {
  return el;
}

// CHECK-LABEL:   func.func @splat_float(
// CHECK-SAME:                           %[[VAL_0:.*]]: f32) -> vector<4xf32>
// CHECK-NEXT:           %[[VAL_1:.*]] = vector.splat %[[VAL_0]] : vector<4xf32>
// CHECK-NEXT:           return %[[VAL_1]] : vector<4xf32>
// CHECK-NEXT:         }

// CHECK-LLVM-LABEL:    define <4 x float> @splat_float(
// CHECK-LLVM-SAME:                                     float %[[VAL_0:.*]]) {
// CHECK-LLVM-NEXT:       %[[VAL_1:.*]] = insertelement <4 x float> undef, float %[[VAL_0]], i32 0
// CHECK-LLVM-NEXT:       %[[VAL_2:.*]] = shufflevector <4 x float> %[[VAL_1]], <4 x float> undef, <4 x i32> zeroinitializer
// CHECK-LLVM-NEXT:       ret <4 x float> %[[VAL_2]]
// CHECK-LLVM-NEXT:     }

float_t_vec splat_float(float el) {
  return el;
}
