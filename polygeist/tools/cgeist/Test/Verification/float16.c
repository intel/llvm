// RUN: cgeist %s --function=* -S -march=x86-64         2>&1 | FileCheck %s --check-prefix=CHECK-EXTEND
// RUN: cgeist %s --function=* -S -march=sapphirerapids 2>&1 | FileCheck %s --check-prefix=CHECK-NATIVE

// COM: sapphirerapids supports _Float16 natively

// CHECK-NATIVE-NOT: warning: Experimental usage of _Float16.

// CHECK-EXTEND-LABEL:  func.func @type(%arg0: f16) -> f16
// CHECK-EXTEND-NEXT:     return %arg0 : f16
// CHECK-EXTEND-NEXT:   }

// CHECK-NATIVE-LABEL:  func.func @type(%arg0: f16) -> f16
// CHECK-NATIVE-NEXT:     return %arg0 : f16
// CHECK-NATIVE-NEXT:   }

_Float16 type(_Float16 arg) {
  return arg;
}

// CHECK-EXTEND-LABEL:  func.func @arith(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16, %arg4: f16) -> f16
// CHECK-EXTEND-NEXT:     %[[EXT0:.*]] = arith.extf %arg0 : f16 to f32
// CHECK-EXTEND-NEXT:     %[[EXT1:.*]] = arith.extf %arg1 : f16 to f32
// CHECK-EXTEND-NEXT:     %[[ADD:.*]] = arith.addf %[[EXT0]], %[[EXT1]] : f32
// CHECK-EXTEND-NEXT:     %[[EXT2:.*]] = arith.extf %arg2 : f16 to f32
// CHECK-EXTEND-NEXT:     %[[EXT3:.*]] = arith.extf %arg3 : f16 to f32
// CHECK-EXTEND-NEXT:     %[[NEG:.*]] = arith.negf %[[EXT3]] : f32
// CHECK-EXTEND-NEXT:     %[[MUL:.*]] = arith.mulf %[[EXT2]], %[[NEG]] : f32
// CHECK-EXTEND-NEXT:     %[[EXT4:.*]] = arith.extf %arg4 : f16 to f32
// CHECK-EXTEND-NEXT:     %[[DIV:.*]] = arith.divf %[[MUL]], %[[EXT4]] : f32
// CHECK-EXTEND-NEXT:     %[[SUB:.*]] = arith.subf %[[ADD]], %[[DIV]] : f32
// CHECK-EXTEND-NEXT:     %[[RES:.*]] = arith.truncf %[[SUB]] : f32 to f16
// CHECK-EXTEND-NEXT:     return %[[RES]] : f16
// CHECK-EXTEND-NEXT:   }

// CHECK-NATIVE-LABEL:  func.func @arith(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16, %arg4: f16) -> f16
// CHECK-NATIVE-NEXT:     %[[ADD:.*]] = arith.addf %arg0, %arg1 : f16
// CHECK-NATIVE-NEXT:     %[[NEG:.*]] = arith.negf %arg3 : f16
// CHECK-NATIVE-NEXT:     %[[MUL:.*]] = arith.mulf %arg2, %[[NEG]] : f16
// CHECK-NATIVE-NEXT:     %[[DIV:.*]] = arith.divf %[[MUL]], %arg4 : f16
// CHECK-NATIVE-NEXT:     %[[SUB:.*]] = arith.subf %[[ADD]], %[[DIV]] : f16
// CHECK-NATIVE-NEXT:     return %[[SUB]] : f16
// CHECK-NATIVE-NEXT:   }

_Float16 arith(_Float16 a,
	       _Float16 b,
	       _Float16 c,
	       _Float16 d,
	       _Float16 e) {
  return (+a) + b - c * (-d) / e;
}

// CHECK-EXTEND-LABEL:  func.func @compound_assign(%arg0: memref<?xf16>, %arg1: f16)
// CHECK-EXTEND-NEXT:     %[[EXT1:.*]] = arith.extf %arg1 : f16 to f32
// CHECK-EXTEND-NEXT:     %[[ORIG:.*]] = affine.load %arg0[0] : memref<?xf16>
// CHECK-EXTEND-NEXT:     %[[EXTORIG:.*]] = arith.extf %[[ORIG]] : f16 to f32
// CHECK-EXTEND-NEXT:     %[[ADD:.*]] = arith.addf %[[EXTORIG]], %[[EXT1]] : f32
// CHECK-EXTEND-NEXT:     %[[RES:.*]] = arith.truncf %[[ADD]] : f32 to f16
// CHECK-EXTEND-NEXT:     affine.store %[[RES]], %arg0[0] : memref<?xf16>
// CHECK-EXTEND-NEXT:     return
// CHECK-EXTEND-NEXT:   }

// CHECK-NATIVE-LABEL:  func.func @compound_assign(%arg0: memref<?xf16>, %arg1: f16)
// CHECK-NATIVE-NEXT:     %[[ORIG:.*]] = affine.load %arg0[0] : memref<?xf16>
// CHECK-NATIVE-NEXT:     %[[ADD:.*]] = arith.addf %[[ORIG]], %arg1 : f16
// CHECK-NATIVE-NEXT:     affine.store %[[ADD]], %arg0[0] : memref<?xf16>
// CHECK-NATIVE-NEXT:     return
// CHECK-NATIVE-NEXT:   }

void compound_assign(_Float16 *a, _Float16 b) {
  *a += b;
}
