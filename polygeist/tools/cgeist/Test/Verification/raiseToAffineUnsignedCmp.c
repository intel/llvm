// RUN: cgeist %s --function=matmul --raise-scf-to-affine -S | FileCheck %s

void matmul(float A[100][200], float B[200][300], float C[100][300]) {
  int i, j, k;

  // CHECK: affine.for
  for (i = 0; i < 100; i++) {
    // CHECK: affine.for
    for (j = 0; j < 300; j++) {
      // CHECK: affine.store %{{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x300xf32>
      C[i][j] = 0;
      // CHECK: affine.for
      for (k = 0; k < 200; k++) {
        // CHECK: {{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x200xf32>
        // CHECK: {{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x300xf32>
        // CHECK: {{.*}} = arith.mulf
        // CHECK: {{.*}} = affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x300xf32>
        // CHECK: {{.*}} = arith.addf
        // CHECK: affine.store {{.*}}, %{{.*}}[%{{.*}}, %{{.*}}] : memref<?x300xf32>
        C[i][j] += A[i][k] * B[k][j];
      }
    }
  }
}
