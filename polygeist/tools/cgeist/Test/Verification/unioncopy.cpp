// RUN: cgeist %s --function=* -S | FileCheck %s

union S {
	double d;
};

class MyScalar {
 public:
  S v;
  MyScalar(double vv) {
   v.d = vv;
  }
};

void use(double);
void meta() {
	MyScalar alpha_scalar(1.0);
	alpha_scalar = MyScalar(3.0);
	use(alpha_scalar.v.d);
}

// CHECK:   func @_Z4metav() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:     %c1_i64 = arith.constant 1 : i64
// CHECK-DAG:     %[[cst:.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:     %[[cst_0:.+]] = arith.constant 3.000000e+00 : f64
// CHECK-NEXT:     %0 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(f64)>)> : (i64) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %1 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(f64)>)> : (i64) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %2 = llvm.alloca %c1_i64 x !llvm.struct<(struct<(f64)>)> : (i64) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     call @_ZN8MyScalarC1Ed(%2, %[[cst]]) : (!llvm.ptr<struct<(struct<(f64)>)>>, f64) -> ()
// CHECK-NEXT:     call @_ZN8MyScalarC1Ed(%1, %[[cst_0]]) : (!llvm.ptr<struct<(struct<(f64)>)>>, f64) -> ()
// CHECK-NEXT:     %3 = llvm.load %1 : !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     llvm.store %3, %0 : !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %4 = call @_ZN8MyScalaraSEOS_(%2, %0) : (!llvm.ptr<struct<(struct<(f64)>)>>, !llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:     %5 = llvm.getelementptr inbounds %2[0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %6 = llvm.getelementptr inbounds %5[0, 0] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     %7 = llvm.load %6 : !llvm.ptr<f64>
// CHECK-NEXT:     call @_Z3used(%7) : (f64) -> ()
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN8MyScalarC1Ed(%arg0: !llvm.ptr<struct<(struct<(f64)>)>>, %arg1: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr inbounds %0[0, 0] : (!llvm.ptr<struct<(f64)>>) -> !llvm.ptr<f64>
// CHECK-NEXT:     llvm.store %arg1, %1 : !llvm.ptr<f64>
// CHECK-NEXT:     return
// CHECK-NEXT:   }
// CHECK:   func @_ZN8MyScalaraSEOS_(%arg0: !llvm.ptr<struct<(struct<(f64)>)>>, %arg1: !llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(struct<(f64)>)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:     %0 = llvm.getelementptr inbounds %arg0[0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %1 = llvm.getelementptr inbounds %arg1[0, 0] : (!llvm.ptr<struct<(struct<(f64)>)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     %2 = call @_ZN1SaSEOS_(%0, %1) : (!llvm.ptr<struct<(f64)>>, !llvm.ptr<struct<(f64)>>) -> !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:     return %arg0 : !llvm.ptr<struct<(struct<(f64)>)>>
// CHECK-NEXT:   }
// CHECK:   func @_ZN1SaSEOS_(%arg0: !llvm.ptr<struct<(f64)>>, %arg1: !llvm.ptr<struct<(f64)>>) -> !llvm.ptr<struct<(f64)>> attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:     %c8_i64 = arith.constant 8 : i64
// CHECK-DAG:     %false = arith.constant false
// CHECK-NEXT:     %[[i0:.+]] = llvm.bitcast %arg0 : !llvm.ptr<struct<(f64)>> to !llvm.ptr<i8>
// CHECK-NEXT:     %[[i1:.+]] = llvm.bitcast %arg1 : !llvm.ptr<struct<(f64)>> to !llvm.ptr<i8>
// CHECK-NEXT:     "llvm.intr.memcpy"(%[[i0]], %[[i1]], %c8_i64, %false) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
// CHECK-NEXT:     return %arg0 : !llvm.ptr<struct<(f64)>>
// CHECK-NEXT:   }
