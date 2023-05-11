// RUN: polygeist-opt -split-input-file -test-memory-access %s 2>&1 | FileCheck %s


!sycl_id_1 = !sycl.id<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_id_2 = !sycl.id<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_range_1 = !sycl.range<[1], (!sycl.array<[1], (memref<1xi64, 4>)>)>
!sycl_range_2 = !sycl.range<[2], (!sycl.array<[2], (memref<2xi64, 4>)>)>
!sycl_accessor_1_f32_rw_gb = !sycl.accessor<[1, f32, read_write, global_buffer], (!sycl.accessor_impl_device<[1], (!sycl_id_1, !sycl_range_1, !sycl_range_1)>, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_2_f32_rw_gb = !sycl.accessor<[2, f32, read_write, global_buffer], (!sycl.accessor_impl_device<[2], (!sycl_id_2, !sycl_range_2, !sycl_range_2)>, !llvm.struct<(memref<?xf32, 1>)>)>

// COM:  Test 1-dim accessor memory access yielding an identity matrix.
func.func @test1(%arg0 : memref<?x!sycl_accessor_1_f32_rw_gb, 4>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_1>
  %alloca_0 = memref.alloca() : memref<1x!sycl_id_1>  
  %cast = memref.cast %alloca : memref<1x!sycl_id_1> to memref<?x!sycl_id_1>  
  %addrspace_cast = memref.memory_space_cast %alloca : memref<1x!sycl_id_1> to memref<1x!sycl_id_1, 4>  
  %cast_0 = memref.cast %alloca_0 : memref<1x!sycl_id_1> to memref<?x!sycl_id_1>    
  %c1 = arith.constant 1 : index

  affine.for %i = 0 to 64 {
    %idx = arith.muli %i, %c1 : index
    sycl.constructor @id(%addrspace_cast, %idx) {MangledFunctionName = @constr, tag_name = "test13_store1"} : (memref<1x!sycl_id_1, 4>, index)
    %1 = affine.load %alloca[0] : memref<1x!sycl_id_1>  
    affine.store %1, %alloca_0[0] : memref<1x!sycl_id_1>
    %2 = sycl.accessor.subscript %arg0[%cast_0] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<?x!sycl_id_1>], FunctionName = @"operator[]", MangledFunctionName = @subscript, TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_rw_gb, 4>, memref<?x!sycl_id_1>) -> memref<?xf32, 4>
    %3 = affine.load %2[0] : memref<?xf32, 4>
  }
  return
}

// COM: Test 2-dim accessor memory access yielding an identity matrix.
func.func @test2(%arg0 : memref<?x!sycl_accessor_2_f32_rw_gb, 4>) {
  %alloca = memref.alloca() : memref<1x!sycl_id_2>
  %cast = memref.cast %alloca : memref<1x!sycl_id_2> to memref<?x!sycl_id_2>
  %addrspace_cast = memref.memory_space_cast %alloca : memref<1x!sycl_id_2> to memref<1x!sycl_id_2, 4>  
  %c1 = arith.constant 1 : index

  affine.for %i = 0 to 64 {
    %idx = arith.muli %i, %c1 : index
    sycl.constructor @id(%addrspace_cast, %idx) {MangledFunctionName = @constr, tag_name = "test13_store1"} : (memref<1x!sycl_id_2, 4>, index)
    %2 = sycl.accessor.subscript %arg0[%cast] {tag = "test13_sub", ArgumentTypes = [memref<?x!sycl_accessor_2_f32_rw_gb, 4>, memref<?x!sycl_id_1>], FunctionName = @"operator[]", MangledFunctionName = @subscript, TypeName = @accessor} : (memref<?x!sycl_accessor_2_f32_rw_gb, 4>, memref<?x!sycl_id_2>) -> memref<?xf32, 4>
    %3 = affine.load %2[0] : memref<?xf32, 4>
  }
  return
}



// Cannot handle this because the AccessMatrix only takes values and not AffineExpr.
//func.func @test2() {
//  %c1 = arith.constant 1 : index
//  %alloca_1 = memref.alloca() : memref<64x64xi32>
//  %alloca_2 = memref.alloca() : memref<64x64xi32>  

//  %s = arith.addi %c1, %v : index

//  affine.for %i = 0 to 64 {
//    affine.for %j = 0 to 64 {    
     // %idx = affine.apply affine_map<(d0) -> (d0 + 1)> (%i)
//      %load1 = affine.load %alloca_1[%i, %j + symbol(%v)] : memref<64x64xi32>    
//      %load1 = affine.load %alloca_1[2*%i, %j+1] : memref<64x64xi32>    
//      affine.store %load1, %alloca_2[%j, %i] : memref<64x64xi32>
//    }
//  }

//  return
//}
