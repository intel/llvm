// RUN: cgeist %s --function=* -S | FileCheck %s

int compute_tran_temp(int total_iterations, int num_iterations)
{
	float t;
    int src = 1, dst = 0;
	for (t = 0; t < total_iterations; t+=num_iterations) {
            int temp = src;
            src = dst;
            dst = temp;
	}
    return dst;
}

// CHECK:   func @compute_tran_temp(%arg0: i32, %arg1: i32) -> i32
// CHECK-DAG:     %cst = arith.constant 0.000000e+00 : f32
// CHECK-DAG:     %c0_i32 = arith.constant 0 : i32
// CHECK-DAG:     %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:     %0 = arith.sitofp %arg0 : i32 to f32
// CHECK-NEXT:     %1 = arith.sitofp %arg1 : i32 to f32
// CHECK-NEXT:     %2:3 = scf.while (%arg2 = %c0_i32, %arg3 = %c1_i32, %arg4 = %cst) : (i32, i32, f32) -> (i32, i32, f32) {
// CHECK-NEXT:       %3 = arith.cmpf olt, %arg4, %0 : f32
// CHECK-NEXT:       scf.condition(%3) %arg2, %arg3, %arg4 : i32, i32, f32
// CHECK-NEXT:     } do {
// CHECK-NEXT:     ^bb0(%arg2: i32, %arg3: i32, %arg4: f32):
// CHECK-NEXT:       %3 = arith.addf %arg4, %1 : f32
// CHECK-NEXT:       scf.yield %arg3, %arg2, %3 : i32, i32, f32
// CHECK-NEXT:     }
// CHECK-NEXT:     return %2#0 : i32
// CHECK-NEXT:   }
