// RUN: polygeist-opt --function-specialization="relaxed-aliasing=false" %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_f32_w_gb = !sycl.accessor<[1, f32, write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_1_f32_r_gb = !sycl.accessor<[1, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_1_i32_r_gb = !sycl.accessor<[1, i32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>
gpu.module @device_func {
  // COM: This function is a candidate, check that it is transformed correctly.
  // CHECK-LABEL: func.func private @callee1.specialized(%arg0: memref<?x!sycl_accessor_1_f32_r_gb> {sycl.inner.disjoint}, %arg1: memref<?x!sycl_accessor_1_f32_w_gb> {sycl.inner.disjoint})
  // CHECK-LABEL: func.func private @callee1(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>)
  // CHECK-LABEL: gpu.func @caller1() kernel
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1x!sycl_accessor_1_f32_r_gb>
  // CHECK-NEXT:    %cast = memref.cast %alloca : memref<1x!sycl_accessor_1_f32_r_gb> to memref<?x!sycl_accessor_1_f32_r_gb>
  // CHECK-NEXT:    %alloca_0 = memref.alloca() : memref<1x!sycl_accessor_1_f32_w_gb>
  // CHECK-NEXT:    %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_accessor_1_f32_w_gb> to memref<?x!sycl_accessor_1_f32_w_gb>

  // COM: Obtain a pointer to the beginning of the first accessor.
  // CHECK-NEXT:    %alloca_2 = memref.alloca() : memref<1x!sycl_id_1_>
  // CHECK-NEXT:    %c0 = arith.constant 0 : index
  // CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT:    %0 = sycl.id.get %alloca_2[%c0_i32] {ArgumentTypes = [memref<1x!sycl_id_1_>, i32], FunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_1_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    memref.store %c0, %0[%c0] : memref<?xindex>
  // CHECK-NEXT:    [[ACC1_BEGIN:%.*]] = sycl.accessor.subscript %cast[%alloca_2] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_r_gb>, memref<1x!sycl_id_1_>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>

  // COM: Obtain a pointer to the end of the first accessor.
  // CHECK-NEXT:    %2 = sycl.accessor.get_range(%cast) {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_r_gb>], FunctionName = @get_range, TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_r_gb>) -> !sycl_range_1_
  // CHECK-NEXT:    %alloca_3 = memref.alloca() : memref<1x!sycl_range_1_>
  // CHECK-NEXT:    %c0_4 = arith.constant 0 : index
  // CHECK-NEXT:    memref.store %2, %alloca_3[%c0_4] : memref<1x!sycl_range_1_>
  // CHECK-NEXT:    %alloca_5 = memref.alloca() : memref<1x!sycl_id_1_>
  // CHECK-NEXT:    %c1 = arith.constant 1 : index
  // CHECK-NEXT:    %c0_i32_6 = arith.constant 0 : i32
  // CHECK-NEXT:    %3 = sycl.id.get %alloca_5[%c0_i32_6] {ArgumentTypes = [memref<1x!sycl_id_1_>, i32], FunctionName = @"operator[]", TypeName = @id} : (memref<1x!sycl_id_1_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    %c0_i32_7 = arith.constant 0 : i32
  // CHECK-NEXT:    %4 = sycl.range.get %alloca_3[%c0_i32_7] {ArgumentTypes = [memref<1x!sycl_range_1_>, i32], FunctionName = @get, TypeName = @range} : (memref<1x!sycl_range_1_>, i32) -> index
  // CHECK-NEXT:    memref.store %4, %3[%c0_4] : memref<?xindex>
  // CHECK-NEXT:    [[ACC1_END:%.*]] = sycl.accessor.subscript %cast[%alloca_5] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_r_gb>, memref<1x!sycl_id_1_>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>

  // COM: Version with condition: [[ACC1_END]] <= [[ACC2_BEGIN]] || [[ACC1_BEGIN]] >= [[ACC2_END]].
  // CHECK:         [[ACC2_BEGIN:%.*]] = sycl.accessor.subscript %cast_1[{{.*}}] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_w_gb>, memref<1x!sycl_id_1_>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_w_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>
  // CHECK:         [[ACC2_END:%.*]] = sycl.accessor.subscript %cast_1[{{.*}}] {ArgumentTypes = [memref<?x!sycl_accessor_1_f32_w_gb>, memref<1x!sycl_id_1_>], FunctionName = @"operator[]", TypeName = @accessor} : (memref<?x!sycl_accessor_1_f32_w_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>
  // CHECK-DAG:     [[ACC1_END_PTR:%.*]] = "polygeist.memref2pointer"([[ACC1_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC2_BEGIN_PTR:%.*]]  = "polygeist.memref2pointer"([[ACC2_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %14 = llvm.icmp "ule" [[ACC1_END_PTR]], [[ACC2_BEGIN_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC1_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ACC1_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC2_END_PTR:%.*]] = "polygeist.memref2pointer"([[ACC2_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %17 = llvm.icmp "uge" [[ACC1_BEGIN_PTR]], [[ACC2_END_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %18 = arith.ori %14, %17 : i1
  // CHECK-NEXT:    scf.if %18 {
  // CHECK-NEXT:      func.call @callee1.specialized(%cast, %cast_1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      func.call @callee1(%cast, %cast_1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    }
  func.func private @callee1(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) {
    return
  }
  gpu.func @caller1() kernel {
    %alloca = memref.alloca() : memref<1x!sycl_accessor_1_f32_r_gb>
    %cast = memref.cast %alloca : memref<1x!sycl_accessor_1_f32_r_gb> to memref<?x!sycl_accessor_1_f32_r_gb>
    %alloca_0 = memref.alloca() : memref<1x!sycl_accessor_1_f32_w_gb>
    %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_accessor_1_f32_w_gb> to memref<?x!sycl_accessor_1_f32_w_gb>
    func.call @callee1(%cast, %cast_1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
    gpu.return
  }

  // COM: No need to version as the accessor types are different.
  // CHECK-LABEL: func.func private @callee2.specialized(%arg0: memref<?x!sycl_accessor_1_i32_r_gb> {sycl.inner.disjoint}, %arg1: memref<?x!sycl_accessor_1_f32_w_gb> {sycl.inner.disjoint}) {
  // CHECK-LABEL: func.func private @callee2(%arg0: memref<?x!sycl_accessor_1_i32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) {
  // CHECK-LABEL: gpu.func @caller2() kernel {
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1x!sycl_accessor_1_i32_r_gb>
  // CHECK-NEXT:    %cast = memref.cast %alloca : memref<1x!sycl_accessor_1_i32_r_gb> to memref<?x!sycl_accessor_1_i32_r_gb>
  // CHECK-NEXT:    %alloca_0 = memref.alloca() : memref<1x!sycl_accessor_1_f32_w_gb>
  // CHECK-NEXT:    %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_accessor_1_f32_w_gb> to memref<?x!sycl_accessor_1_f32_w_gb>
  // CHECK-NEXT:    func.call @callee2.specialized(%cast, %cast_1) : (memref<?x!sycl_accessor_1_i32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    gpu.return
  // CHECK-NEXT:  }
  func.func private @callee2(%arg0: memref<?x!sycl_accessor_1_i32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) {
    return
  }
  gpu.func @caller2() kernel {
    %alloca = memref.alloca() : memref<1x!sycl_accessor_1_i32_r_gb>
    %cast = memref.cast %alloca : memref<1x!sycl_accessor_1_i32_r_gb> to memref<?x!sycl_accessor_1_i32_r_gb>
    %alloca_0 = memref.alloca() : memref<1x!sycl_accessor_1_f32_w_gb>
    %cast_1 = memref.cast %alloca_0 : memref<1x!sycl_accessor_1_f32_w_gb> to memref<?x!sycl_accessor_1_f32_w_gb>
    func.call @callee2(%cast, %cast_1) : (memref<?x!sycl_accessor_1_i32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
    gpu.return
  }
}
