// RUN: polygeist-opt --convert-polygeist-to-llvm --split-input-file %s | FileCheck %s

// CHECK-LABEL: @test_1
// CHECK: [[ZERO:%.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK: [[GEP:%.*]] = llvm.getelementptr {{.*}}[[[ZERO]], 0] : (!llvm.ptr<struct<([[SYCLIDSTRUCT:struct<"class.cl::sycl::id.1"]], {{.*}} -> !llvm.ptr<[[SYCLIDSTRUCT]], {{.*}}
// CHECK: [[MEMREF:%.*]] = llvm.mlir.undef : !llvm.struct<(ptr<[[SYCLIDSTRUCT]], {{.*}}
// CHECK: {{.*}} = llvm.insertvalue [[GEP]], [[MEMREF]][0] : !llvm.struct<(ptr<[[SYCLIDSTRUCT]], {{.*}}

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

func.func @test_2(%arg0 : memref<?x!sycl_accessor_1_i32_read_write_global_buffer>) -> memref<?x!sycl_accessor_impl_device_1_> { 
  %c0 = arith.constant 0 : index
  %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!sycl_accessor_1_i32_read_write_global_buffer>, index) -> memref<?x!sycl_accessor_impl_device_1_>
  return %0 : memref<?x!sycl_accessor_impl_device_1_>
}

// -----
