// RUN: sycl-mlir-opt %s | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s --mlir-print-op-generic | sycl-mlir-opt | FileCheck %s
// RUN: sycl-mlir-opt %s -canonicalize | FileCheck %s --check-prefix=FOLD

!sycl_half = !sycl.half<(f16)>

// CHECK-LABEL: test_half
func.func @test_half(%arg0 : f16, %arg1 : !sycl_half) -> (!sycl_half, f16) {
  // CHECK: %{{.*}} = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  %0 = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  // CHECK: %{{.*}} = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  %1 = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  
  return %0, %1 : !sycl_half, f16
}

// FOLD-LABEL: test_folder_half
func.func @test_folder_half(%arg0 : f16, %arg1 : !sycl_half) -> (!sycl_half, f16) {
  %0 = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  %1 = sycl.mlir.unwrap %0 : !sycl_half to f16
  // FOLD: %[[wrap:.*]] = sycl.mlir.wrap %arg0 : f16 to !sycl_half
  %2 = sycl.mlir.wrap %1 : f16 to !sycl_half

  %3 = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  %4 = sycl.mlir.wrap %3 : f16 to !sycl_half
  // FOLD: %[[unwrap:.*]] = sycl.mlir.unwrap %arg1 : !sycl_half to f16
  %5 = sycl.mlir.unwrap %4 : !sycl_half to f16
  
  // FOLD: return %[[wrap]], %[[unwrap]] : !sycl_half, f16
  return %2, %5 : !sycl_half, f16
}

!sycl_vec_i32_4_ = !sycl.vec<[i32, 4], (vector<4xi32>)>

// CHECK-LABEL: test_vector_of_native
func.func @test_vector_of_native(%arg0 : vector<4xi32>, %arg1 : !sycl_vec_i32_4_) -> (!sycl_vec_i32_4_, vector<4xi32>) {
  // CHECK: %{{.*}} = sycl.mlir.wrap %arg0 : vector<4xi32> to !sycl_vec_i32_4_
  %0 = sycl.mlir.wrap %arg0 : vector<4xi32> to !sycl_vec_i32_4_
  // CHECK: %{{.*}} = sycl.mlir.unwrap %arg1 : !sycl_vec_i32_4_ to vector<4xi32>
  %1 = sycl.mlir.unwrap %arg1 : !sycl_vec_i32_4_ to vector<4xi32>
  
  return %0, %1 : !sycl_vec_i32_4_, vector<4xi32>
}

// FOLD-LABEL: test_folder_vector_of_native
func.func @test_folder_vector_of_native(%arg0 : vector<4xi32>, %arg1 : !sycl_vec_i32_4_) -> (!sycl_vec_i32_4_, vector<4xi32>) {
  %0 = sycl.mlir.wrap %arg0 : vector<4xi32> to !sycl_vec_i32_4_
  %1 = sycl.mlir.unwrap %0 : !sycl_vec_i32_4_ to vector<4xi32>
  // FOLD: %[[wrap:.*]] = sycl.mlir.wrap %arg0 : vector<4xi32> to !sycl_vec_i32_4_
  %2 = sycl.mlir.wrap %1 : vector<4xi32> to !sycl_vec_i32_4_

  %3 = sycl.mlir.unwrap %arg1 : !sycl_vec_i32_4_ to vector<4xi32>
  %4 = sycl.mlir.wrap %3 : vector<4xi32> to !sycl_vec_i32_4_
  // FOLD: %[[unwrap:.*]] = sycl.mlir.unwrap %arg1 : !sycl_vec_i32_4_ to vector<4xi32>
  %5 = sycl.mlir.unwrap %4 : !sycl_vec_i32_4_ to vector<4xi32>
  
  // FOLD: return %[[wrap]], %[[unwrap]] : !sycl_vec_i32_4_, vector<4xi32>
  return %2, %5 : !sycl_vec_i32_4_, vector<4xi32>
}

!sycl_vec_sycl_half_2_ = !sycl.vec<[!sycl_half, 2], (vector<2xf16>)>

// CHECK-LABEL: test_vector_of_half
func.func @test_vector_of_half(%arg0 : vector<2xf16>, %arg1 : !sycl_vec_sycl_half_2_) -> (!sycl_vec_sycl_half_2_, vector<2xf16>) {
  // CHECK: %{{.*}} = sycl.mlir.wrap %arg0 : vector<2xf16> to !sycl_vec_sycl_half_2_
  %0 = sycl.mlir.wrap %arg0 : vector<2xf16> to !sycl_vec_sycl_half_2_
  // CHECK: %{{.*}} = sycl.mlir.unwrap %arg1 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  %1 = sycl.mlir.unwrap %arg1 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  
  return %0, %1 : !sycl_vec_sycl_half_2_, vector<2xf16>
}

// FOLD-LABEL: test_folder_vector_of_half
func.func @test_folder_vector_of_half(%arg0 : vector<2xf16>, %arg1 : !sycl_vec_sycl_half_2_) -> (!sycl_vec_sycl_half_2_, vector<2xf16>) {
  %0 = sycl.mlir.wrap %arg0 : vector<2xf16> to !sycl_vec_sycl_half_2_
  %1 = sycl.mlir.unwrap %0 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  // FOLD: %[[wrap:.*]] = sycl.mlir.wrap %arg0 : vector<2xf16> to !sycl_vec_sycl_half_2_
  %2 = sycl.mlir.wrap %1 : vector<2xf16> to !sycl_vec_sycl_half_2_

  %3 = sycl.mlir.unwrap %arg1 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  %4 = sycl.mlir.wrap %3 : vector<2xf16> to !sycl_vec_sycl_half_2_
  // FOLD: %[[unwrap:.*]] = sycl.mlir.unwrap %arg1 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  %5 = sycl.mlir.unwrap %4 : !sycl_vec_sycl_half_2_ to vector<2xf16>
  
  // FOLD: return %[[wrap]], %[[unwrap]] : !sycl_vec_sycl_half_2_, vector<2xf16>
  return %2, %5 : !sycl_vec_sycl_half_2_, vector<2xf16>
}
