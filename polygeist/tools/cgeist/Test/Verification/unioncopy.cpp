// RUN: cgeist  %s --function=* -S | FileCheck %s

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

// CHECK-LABEL:   func.func @_Z4metav() attributes {llvm.linkage = #llvm.linkage<external>} {
// CHECK-DAG:       %[[VAL_0:.*]] = arith.constant 3.000000e+00 : f64
// CHECK-DAG:       %[[VAL_1:.*]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 1 : i64
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(f64)>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_4:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(f64)>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_5:.*]] = llvm.alloca %[[VAL_2]] x !llvm.struct<(struct<(f64)>)> : (i64) -> !llvm.ptr
// CHECK-NEXT:      call @_ZN8MyScalarC1Ed(%[[VAL_5]], %[[VAL_1]]) : (!llvm.ptr, f64) -> ()
// CHECK-NEXT:      call @_ZN8MyScalarC1Ed(%[[VAL_4]], %[[VAL_0]]) : (!llvm.ptr, f64) -> ()
// CHECK-NEXT:      %[[VAL_6:.*]] = llvm.load %[[VAL_4]] : !llvm.ptr -> !llvm.struct<(struct<(f64)>)>
// CHECK-NEXT:      llvm.store %[[VAL_6]], %[[VAL_3]] : !llvm.struct<(struct<(f64)>)>, !llvm.ptr
// CHECK-NEXT:      %[[VAL_7:.*]] = call @_ZN8MyScalaraSEOS_(%[[VAL_5]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      %[[VAL_8:.*]] = llvm.getelementptr inbounds %[[VAL_5]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64)>)>
// CHECK-NEXT:      %[[VAL_9:.*]] = llvm.getelementptr inbounds %[[VAL_8]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64)>
// CHECK-NEXT:      %[[VAL_10:.*]] = llvm.load %[[VAL_9]] : !llvm.ptr -> f64
// CHECK-NEXT:      call @_Z3used(%[[VAL_10]]) : (f64) -> ()
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN8MyScalarC1Ed(
// CHECK-SAME:                                %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                %[[VAL_1:.*]]: f64) attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64)>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_2]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(f64)>
// CHECK-NEXT:      llvm.store %[[VAL_1]], %[[VAL_3]] : f64, !llvm.ptr
// CHECK-NEXT:      return
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN8MyScalaraSEOS_(
// CHECK-SAME:                                  %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                                  %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-NEXT:      %[[VAL_2:.*]] = llvm.getelementptr inbounds %[[VAL_0]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64)>)>
// CHECK-NEXT:      %[[VAL_3:.*]] = llvm.getelementptr inbounds %[[VAL_1]][0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(struct<(f64)>)>
// CHECK-NEXT:      %[[VAL_4:.*]] = call @_ZN1SaSEOS_(%[[VAL_2]], %[[VAL_3]]) : (!llvm.ptr, !llvm.ptr) -> !llvm.ptr
// CHECK-NEXT:      return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }

// CHECK-LABEL:   func.func @_ZN1SaSEOS_(
// CHECK-SAME:                           %[[VAL_0:.*]]: !llvm.ptr,
// CHECK-SAME:                           %[[VAL_1:.*]]: !llvm.ptr) -> !llvm.ptr attributes {llvm.linkage = #llvm.linkage<linkonce_odr>} {
// CHECK-DAG:       %[[VAL_2:.*]] = arith.constant 8 : i64
// CHECK-NEXT:      "llvm.intr.memcpy"(%[[VAL_0]], %[[VAL_1]], %[[VAL_2]]) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()
// CHECK-NEXT:      return %[[VAL_0]] : !llvm.ptr
// CHECK-NEXT:    }
