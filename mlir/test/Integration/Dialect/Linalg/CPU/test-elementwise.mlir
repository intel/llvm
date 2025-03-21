// RUN: mlir-opt %s -convert-elementwise-to-linalg \
// RUN: -one-shot-bufferize="bufferize-function-boundaries" \
// RUN: -canonicalize -buffer-deallocation-pipeline -convert-bufferization-to-memref -convert-linalg-to-loops \
// RUN: -convert-scf-to-cf -convert-arith-to-llvm -convert-cf-to-llvm --finalize-memref-to-llvm \
// RUN: -convert-func-to-llvm -reconcile-unrealized-casts | \
// RUN: mlir-runner -e main -entry-point-result=void \
// RUN:   -shared-libs=%mlir_runner_utils \
// RUN: | FileCheck %s

func.func @main() {
  %a = arith.constant dense<[1.0, 2.0, 3.0]> : tensor<3xf32>
  %b = arith.constant dense<[10.0, 20.0, 30.0]> : tensor<3xf32>

  %addf = arith.addf %a, %b : tensor<3xf32>
  %addf_unranked = tensor.cast %addf : tensor<3xf32> to tensor<*xf32>
  call @printMemrefF32(%addf_unranked) : (tensor<*xf32>) -> ()
  // CHECK: Unranked Memref base@ = {{.*}} rank = 1 offset = 0 sizes = [3] strides = [1] data =
  // CHECK-NEXT: [11,  22,  33]

  return
}

func.func private @printMemrefF32(%ptr : tensor<*xf32>)
