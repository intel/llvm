// RUN: polygeist-opt --canonicalize-scf-for --split-input-file %s | FileCheck %s

module {
  func.func private @cmp() -> i1

  func.func @_Z4div_Pi(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) {
	  %c0_i32 = arith.constant 0 : i32
	  %c1_i32 = arith.constant 1 : i32
	  %c3_i64 = arith.constant 3 : index
	  %1:3 = scf.while (%arg3 = %c0_i32) : (i32) -> (i32, index, index) {
		%2 = arith.index_cast %arg3 : i32 to index
		%3 = arith.addi %2, %c3_i64 : index
		%5 = func.call @cmp() : () -> i1
		scf.condition(%5) %arg3, %3, %2 : i32, index, index
	  } do {
	  ^bb0(%arg3: i32, %arg4: index, %arg5: index):  
		%parg3 = arith.addi %arg3, %c1_i32 : i32
		%3 = memref.load %arg0[%arg5] : memref<?xi32>
		memref.store %3, %arg1[%arg4] : memref<?xi32>
		scf.yield %parg3 : i32
	  }
	  return
  }

}


// CHECK: func.func @_Z4div_Pi(%arg0: memref<?xi32>, %arg1: memref<?xi32>, %arg2: i32) {
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-DAG:     %c3 = arith.constant 3 : index
// CHECK-NEXT:    %0 = scf.while (%arg3 = %c0_i32) : (i32) -> i32 {
// CHECK-NEXT:       %1 = func.call @cmp() : () -> i1
// CHECK-NEXT:       scf.condition(%1) %arg3 : i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg3: i32):  
// CHECK-NEXT:       %1 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:       %2 = arith.index_cast %arg3 : i32 to index
// CHECK-NEXT:       %3 = arith.addi %1, %c3 : index
// CHECK-NEXT:       %4 = arith.addi %arg3, %c1_i32 : i32
// CHECK-NEXT:       %5 = memref.load %arg0[%2] : memref<?xi32>
// CHECK-NEXT:       memref.store %5, %arg1[%3] : memref<?xi32>
// CHECK-NEXT:       scf.yield %4 : i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return
// CHECK-NEXT:   }

// -----

module {
  func.func @gcd(%arg0: i32, %arg1: i32) -> i32 {
    %c0_i32 = arith.constant 0 : i32
    %0:2 = scf.while (%arg2 = %arg1, %arg3 = %arg0) : (i32, i32) -> (i32, i32) {
      %1 = arith.cmpi sgt, %arg2, %c0_i32 : i32
      %2:2 = scf.if %1 -> (i32, i32) {
        %3 = arith.remsi %arg3, %arg2 : i32
        scf.yield %3, %arg2 : i32, i32
      } else {
        scf.yield %arg2, %arg3 : i32, i32
      }
      scf.condition(%1) %2#0, %2#1 : i32, i32
    } do {
    ^bb0(%arg2: i32, %arg3: i32):  
      scf.yield %arg2, %arg3 : i32, i32
    }
    return %0#1 : i32
  }
}

// CHECK:   func.func @gcd(%arg0: i32, %arg1: i32) -> i32 {
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0:2 = scf.while (%arg2 = %arg1, %arg3 = %arg0) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:       %1 = arith.cmpi sgt, %arg2, %c0_i32 : i32
// CHECK-NEXT:       scf.condition(%1) %arg3, %arg2 : i32, i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg2: i32, %arg3: i32):  
// CHECK-NEXT:       %1 = arith.remsi %arg2, %arg3 : i32
// CHECK-NEXT:       scf.yield %1, %arg3 : i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#0 : i32
// CHECK-NEXT:   }

// -----

module  {
  func.func @runHisto(%arg0: i32, %arg1: i32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = scf.while (%arg2 = %c0_i32) : (i32) -> i32 {
      %1 = arith.cmpi slt, %arg2, %arg0 : i32
      %2 = scf.if %1 -> (i32) {
        func.call @histo_kernel() : () -> ()
        %3 = arith.muli %arg1, %c2_i32 : i32
        %4 = arith.addi %arg2, %3 : i32
        scf.yield %4 : i32
      } else {
        scf.yield %arg2 : i32
      }
      scf.condition(%1) %2 : i32
    } do {
    ^bb0(%arg2: i32):  
      scf.yield %arg2 : i32
    }
    return %c0_i32 : i32
  }
  func.func private @histo_kernel() attributes {llvm.linkage = #llvm.linkage<external>}
}

// CHECK:   func.func @runHisto(%arg0: i32, %arg1: i32) -> i32
// CHECK-DAG:     %c2_i32 = arith.constant 2 : i32
// CHECK-DAG:     %c0 = arith.constant 0 : index
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %0 = arith.muli %arg1, %c2_i32 : i32
// CHECK-NEXT:     %1 = arith.index_cast %arg0 : i32 to index
// CHECK-NEXT:     %2 = arith.index_cast %0 : i32 to index
// CHECK-NEXT:     scf.for %arg2 = %c0 to %1 step %2 {
// CHECK-NEXT:       func.call @histo_kernel() : () -> ()
// CHECK-NEXT:     }
// CHECK-NEXT:     return %c0_i32 : i32
// CHECK-NEXT:   }

// -----

module {
  func.func @compute_tran_temp(%1: f32, %4: f32) -> i32 attributes {llvm.linkage = #llvm.linkage<external>} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant 0.000000e+00 : f32
    %0:3 = scf.while (%arg2 = %cst, %arg3 = %c0_i32, %arg4 = %c1_i32) : (f32, i32, i32) -> (f32, i32, i32) {
      %2 = arith.cmpf ult, %arg2, %1 : f32
      %3:3 = scf.if %2 -> (f32, i32, i32) {
        %5 = arith.addf %arg2, %4 : f32
        scf.yield %5, %arg4, %arg3 : f32, i32, i32
      } else {
        scf.yield %arg2, %arg3, %arg4 : f32, i32, i32
      }
      scf.condition(%2) %3#0, %3#1, %3#2 : f32, i32, i32
    } do {
    ^bb0(%arg2: f32, %arg3: i32, %arg4: i32):  
      scf.yield %arg2, %arg3, %arg4 : f32, i32, i32
    }
    return %0#1 : i32
  }
}

// CHECK:   func.func @compute_tran_temp(%arg0: f32, %arg1: f32) -> i32 
// CHECK-NEXT:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-NEXT:     %0:3 = scf.while (%arg2 = %cst, %arg3 = %c0_i32, %arg4 = %c1_i32) : (f32, i32, i32) -> (i32, f32, i32) {
// CHECK-NEXT:       %1 = arith.cmpf ult, %arg2, %arg0 : f32
// CHECK-NEXT:       scf.condition(%1) %arg3, %arg2, %arg4 : i32, f32, i32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg2: i32, %arg3: f32, %arg4: i32):  
// CHECK-NEXT:       %1 = arith.addf %arg3, %arg1 : f32
// CHECK-NEXT:       scf.yield %1, %arg4, %arg2 : f32, i32, i32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %0#0 : i32
// CHECK-NEXT:   }

// -----
  
module {
  func.func @_Z8lud_cudaPfi(%arg0: memref<?xf32>, %arg1: index, %0 : memref<16x16xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c16 = arith.constant 16 : index
    %2 = scf.for %arg2 = %c0 to %c16 step %c1 iter_args(%arg3 = %c0) -> (index) {
      %4 = memref.load %arg0[%arg3] : memref<?xf32>
      memref.store %4, %0[%arg2, %c0] : memref<16x16xf32>
      %5 = arith.addi %arg3, %arg1 : index
      scf.yield %5 : index
    }
    return
  }
}
// CHECK:     scf.for %arg3 = %c0 to %c16 step %c1 {
// CHECK-NEXT:       %0 = arith.muli %arg3, %arg1 : index
// CHECK-NEXT:       %1 = memref.load %arg0[%0] : memref<?xf32>
// CHECK-NEXT:       memref.store %1, %arg2[%arg3, %c0] : memref<16x16xf32>
// CHECK-NEXT:     }
