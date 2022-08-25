// RUN: polygeist-opt --convert-polygeist-to-llvm %s | FileCheck %s

// CHECK: [[GEP:%.*]] = llvm.getelementptr {{.*}} : (!llvm.ptr<struct<([[SYCLIDSTRUCT:struct<"class.cl::sycl::id.1"]], {{.*}} -> !llvm.ptr<[[SYCLIDSTRUCT]], {{.*}}
// CHECK: [[MEMREF:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<[[SYCLIDSTRUCT]], {{.*}}
// CHECK: {{.*}} = llvm.insertvalue [[GEP]], [[MEMREF]][0] : !llvm.struct<(ptr<[[SYCLIDSTRUCT]], {{.*}}

module {
  func.func @test(%arg0: memref<?x!llvm.struct<(!sycl.id<1>)>>) -> memref<?x!sycl.id<1>> {
    %c0 = arith.constant 0 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(!sycl.id<1>)>>, index) -> memref<?x!sycl.id<1>>
    return %0 : memref<?x!sycl.id<1>>
  }
}
