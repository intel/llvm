// RUN: polygeist-opt --convert-polygeist-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_1
// CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[GEP:%.*]] = llvm.getelementptr %{{.*}}[[[ZERO]], 0] : (!llvm.ptr<struct<([[SYCLIDSTRUCT:struct<"class.cl::sycl::id.1"]], {{.*}} -> !llvm.ptr<[[SYCLIDSTRUCT]], {{.*}}
// CHECK: [[MEMREF:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<[[SYCLIDSTRUCT]], {{.*}}
// CHECK: %{{.*}} = llvm.insertvalue [[GEP]], [[MEMREF]][0] : !llvm.struct<(ptr<[[SYCLIDSTRUCT]], {{.*}}

func.func @test_1(%arg0: memref<?x!llvm.struct<(!sycl.id<1>)>>) -> memref<?x!sycl.id<1>> {
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(!sycl.id<1>)>>, index) -> memref<?x!sycl.id<1>>
  return %0 : memref<?x!sycl.id<1>>
}

// -----

// CHECK-LABEL: @test_2
// CHECK: llvm.return %{{.*}} : !llvm.struct<(ptr<struct<"class.cl::sycl::detail::AccessorImplDevice 

!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl.id<1>, !sycl.range<1>, !sycl.range<1>)>
!sycl_accessor_1_i32_read_write_global_buffer = !sycl.accessor<[1, i32, read_write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<i32, 1>)>)>

func.func @test_2(%arg0: memref<?x!sycl_accessor_1_i32_read_write_global_buffer>) -> memref<?x!sycl_accessor_impl_device_1_> { 
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!sycl_accessor_1_i32_read_write_global_buffer>, index) -> memref<?x!sycl_accessor_impl_device_1_>
  return %0 : memref<?x!sycl_accessor_impl_device_1_>
}

// -----

// CHECK-LABEL: @test_3
// CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[GEP:%.*]] = llvm.getelementptr %{{.*}}[[[ZERO]], 0] : (!llvm.ptr<struct<(i32)>>, i64) -> !llvm.ptr<i32>
// CHECK: [[MEMREF:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: %{{.*}} = llvm.insertvalue [[GEP]], [[MEMREF]][0] : !llvm.struct<(ptr<i32>, ptr<i32>, i64, array<1 x i64>, array<1 x i64>)>

func.func @test_3(%arg0: memref<?x!llvm.struct<(i32)>>) -> memref<?xi32> {
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(i32)>>, index) -> memref<?xi32>
  return %0 : memref<?xi32>
}

// -----

// CHECK-LABEL: @test_4
// CHECK: [[GEP:%.*]] = llvm.getelementptr %{{.*}}[{{.*}}] : (!llvm.ptr<struct<([[IDTYPE:struct<"class.cl::sycl::id.1", \(struct<"class.cl::sycl::detail::array.1", \(array<1 x i64>\)>\)>]])>>, i64) -> !llvm.ptr<struct<([[IDTYPE]])>>
// CHECK: [[MEMREF:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<struct<([[IDTYPE]])>>, ptr<struct<([[IDTYPE]])>>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: [[MEMREF1:%.*]] = llvm.insertvalue {{.*}}, [[MEMREF]][0] : !llvm.struct<(ptr<struct<([[IDTYPE]])>>, ptr<struct<([[IDTYPE]])>>, i64, array<1 x i64>, array<1 x i64>)>
// CHECK: %{{.*}} = llvm.insertvalue [[GEP]], [[MEMREF1]][1] : !llvm.struct<(ptr<struct<([[IDTYPE]])>>, ptr<struct<([[IDTYPE]])>>, i64, array<1 x i64>, array<1 x i64>)>

func.func @test_4(%arg0: memref<1x!llvm.struct<(!sycl.id<1>)>>, %arg1: index) -> memref<?x!llvm.struct<(!sycl.id<1>)>> {
  %0 = "polygeist.subindex"(%arg0, %arg1) : (memref<1x!llvm.struct<(!sycl.id<1>)>>, index) -> memref<?x!llvm.struct<(!sycl.id<1>)>>
  return %0 : memref<?x!llvm.struct<(!sycl.id<1>)>>
}

// -----
