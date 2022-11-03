// RUN: cgeist %s --function=* -S -ffloat16 | FileCheck %s

// CHECK-LABEL:  func.func @type(%arg0: f16) -> f16
// CHECK-NEXT:     return %arg0 : f16
// CHECK-NEXT:   }

_Float16 type(_Float16 arg) {
  return arg;
}

// CHECK-LABEL:  func.func @arith(%arg0: f16, %arg1: f16, %arg2: f16, %arg3: f16, %arg4: f16) -> f16
// CHECK-NEXT:     %[[ADD:.*]] = arith.addf %arg0, %arg1 : f16
// CHECK-NEXT:     %[[NEG:.*]] = arith.negf %arg3 : f16
// CHECK-NEXT:     %[[MUL:.*]] = arith.mulf %arg2, %[[NEG]] : f16
// CHECK-NEXT:     %[[DIV:.*]] = arith.divf %[[MUL]], %arg4 : f16
// CHECK-NEXT:     %[[SUB:.*]] = arith.subf %[[ADD]], %[[DIV]] : f16
// CHECK-NEXT:     return %[[SUB]] : f16
// CHECK-NEXT:   }

_Float16 arith(_Float16 a,
	       _Float16 b,
	       _Float16 c,
	       _Float16 d,
	       _Float16 e) {
  return (+a) + b - c * (-d) / e;
}

// CHECK-LABEL:  func.func @compound_assign(%arg0: memref<?xf16>, %arg1: f16)
// CHECK-NEXT:     %[[ORIG:.*]] = affine.load %arg0[0] : memref<?xf16>
// CHECK-NEXT:     %[[ADD:.*]] = arith.addf %[[ORIG]], %arg1 : f16
// CHECK-NEXT:     affine.store %[[ADD]], %arg0[0] : memref<?xf16>
// CHECK-NEXT:     return
// CHECK-NEXT:   }

void compound_assign(_Float16 *a, _Float16 b) {
  *a += b;
}
