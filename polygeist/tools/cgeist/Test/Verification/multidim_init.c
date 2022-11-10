// RUN: cgeist %s -S --function=* | FileCheck %s

  float glob_A[][4] = {
      {1.0f, 2.0, 3.0, 4.0}, 
      {3.33333f},
      {0.1f, 0.2f, 0.3, 0.4},
  };

float foo(int i, int j) {
  // multiple dims with array fillers
  float A[][4] = {
      {1.0f, 2.0, 3.0, 4.0}, 
      {3.33333f},
      {0.1f, 0.2f, 0.3, 0.4},
  };

  // single dim
  float B[4] = {1.23f};

  float sum = 0.0f;
  // dynamic initialization
  for (int k = 0; k < 3; ++k) {
    float C[2] = {i + k, k - j};
    sum += C[i];
  }

  return glob_A[i][j] + A[i][j] + B[j] + sum;
}
// CHECK: memref.global @glob_A : memref<3x4xf32> = dense{{.*}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00], [3.333330e+00, 3.333330e+00, 3.333330e+00, 3.333330e+00], [1.000000e-01, 2.000000e-01, 3.000000e-01, 4.000000e-01{{.*}} {alignment = 16 : i64}
// CHECK-LABEL: func @foo
// CHECK-DAG: %[[CST3:.*]] = arith.constant 3.33
// CHECK-DAG: %[[CST1_23:.*]] = arith.constant 1.23
// CHECK-DAG: %[[MEM_A:.*]] = memref.alloca() : memref<3x4xf32>
// CHECK-DAG: %[[MEM_B:.*]] = memref.alloca() : memref<4xf32>
// CHECK-DAG: %[[MEM_C:.*]] = memref.alloca() : memref<2xf32>
// CHECK: affine.store %{{.*}}, %[[MEM_A]][0, 0]
// CHECK: affine.store %{{.*}}, %[[MEM_A]][0, 1]
// CHECK: affine.store %{{.*}}, %[[MEM_A]][0, 2]
// CHECK: affine.store %{{.*}}, %[[MEM_A]][0, 3]
// CHECK: affine.store %[[CST3]], %[[MEM_A]][1, 0]
// CHECK: affine.store %[[CST3]], %[[MEM_A]][1, 1]
// CHECK: affine.store %[[CST3]], %[[MEM_A]][1, 2]
// CHECK: affine.store %[[CST3]], %[[MEM_A]][1, 3]
// CHECK: affine.store %{{.*}}, %[[MEM_A]][2, 0]
// CHECK: affine.store %{{.*}}, %[[MEM_A]][2, 1]
// CHECK: affine.store %{{.*}}, %[[MEM_A]][2, 2]
// CHECK: affine.store %{{.*}}, %[[MEM_A]][2, 3]

// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][0]
// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][1]
// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][2]
// CHECK: affine.store %[[CST1_23]], %[[MEM_B]][3]

// CHECK: scf.for
// CHECK: affine.store %{{.*}}, %[[MEM_C]][0]
// CHECK: affine.store %{{.*}}, %[[MEM_C]][1]
