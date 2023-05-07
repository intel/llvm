// RUN: polygeist-opt --kernel-disjoint-specialization="relaxed-aliasing=false" %s | FileCheck %s

!sycl_array_1_ = !sycl.array<[1], (memref<1xi64, 4>)>
!sycl_range_1_ = !sycl.range<[1], (!sycl_array_1_)>
!sycl_id_1_ = !sycl.id<[1], (!sycl_array_1_)>
!sycl_accessor_impl_device_1_ = !sycl.accessor_impl_device<[1], (!sycl_id_1_, !sycl_range_1_, !sycl_range_1_)>
!sycl_accessor_1_f32_w_gb = !sycl.accessor<[1, f32, write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(ptr<1>)>)>
!sycl_accessor_1_f32_r_gb = !sycl.accessor<[1, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_1_i32_r_gb = !sycl.accessor<[1, i32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xi32, 1>)>)>

!sycl_array_2_ = !sycl.array<[2], (memref<1xi64, 4>)>
!sycl_range_2_ = !sycl.range<[2], (!sycl_array_2_)>
!sycl_id_2_ = !sycl.id<[2], (!sycl_array_2_)>
!sycl_accessor_impl_device_2_ = !sycl.accessor_impl_device<[2], (!sycl_id_2_, !sycl_range_2_, !sycl_range_2_)>
!sycl_accessor_2_f32_w_gb = !sycl.accessor<[2, f32, write, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_2_f32_r_gb = !sycl.accessor<[2, f32, read, global_buffer], (!sycl_accessor_impl_device_2_, !llvm.struct<(memref<?xf32, 1>)>)>

!sycl_accessor_0_f32_w_gb = !sycl.accessor<[0, f32, write, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>
!sycl_accessor_0_f32_r_gb = !sycl.accessor<[0, f32, read, global_buffer], (!sycl_accessor_impl_device_1_, !llvm.struct<(memref<?xf32, 1>)>)>

gpu.module @device_func {
  // COM: This function is a candidate, check that it is transformed correctly.
  // CHECK-LABEL: func.func private @callee1.specialized(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_1_f32_r_gb> {sycl.inner.disjoint},
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_1_f32_w_gb> {sycl.inner.disjoint})
  // CHECK-LABEL: func.func private @callee1(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_1_f32_r_gb>,
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_1_f32_w_gb>)
  // CHECK-LABEL: gpu.func @caller1(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) kernel {

  // COM: Obtain a pointer to the beginning of the first accessor.
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1x!sycl_id_1_>
  // CHECK-NEXT:    %c0 = arith.constant 0 : index
  // CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT:    %0 = sycl.id.get %alloca[%c0_i32] : (memref<1x!sycl_id_1_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    memref.store %c0, %0[%c0] : memref<?xindex>
  // CHECK-NEXT:    [[ACC1_BEGIN:%.*]] = sycl.accessor.subscript %arg0[%alloca] : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>

  // COM: Obtain a pointer to the end of the first accessor.
  // CHECK-NEXT:    %2 = sycl.accessor.get_range(%arg0) : (memref<?x!sycl_accessor_1_f32_r_gb>) -> !sycl_range_1_
  // CHECK-NEXT:    %alloca_0 = memref.alloca() : memref<1x!sycl_range_1_>
  // CHECK-NEXT:    %c0_1 = arith.constant 0 : index
  // CHECK-NEXT:    memref.store %2, %alloca_0[%c0_1] : memref<1x!sycl_range_1_>
  // CHECK-NEXT:    %alloca_2 = memref.alloca() : memref<1x!sycl_id_1_>
  // CHECK-NEXT:    %c1 = arith.constant 1 : index
  // CHECK-NEXT:    %c0_i32_3 = arith.constant 0 : i32
  // CHECK-NEXT:    %3 = sycl.id.get %alloca_2[%c0_i32_3] : (memref<1x!sycl_id_1_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    %c0_i32_4 = arith.constant 0 : i32
  // CHECK-NEXT:    %4 = sycl.range.get %alloca_0[%c0_i32_4] : (memref<1x!sycl_range_1_>, i32) -> index
  // CHECK-NEXT:    memref.store %4, %3[%c0_1] : memref<?xindex>
  // CHECK-NEXT:    [[ACC1_END:%.*]] = sycl.accessor.subscript %arg0[%alloca_2] : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>

  // COM: Version with condition: [[ACC1_END]] <= [[ACC2_BEGIN]] || [[ACC1_BEGIN]] >= [[ACC2_END]].
  // CHECK:         [[ACC2_BEGIN:%.*]] = sycl.accessor.subscript %arg1[{{.*}}] : (memref<?x!sycl_accessor_1_f32_w_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>
  // CHECK:         [[ACC2_END:%.*]] = sycl.accessor.subscript %arg1[{{.*}}] : (memref<?x!sycl_accessor_1_f32_w_gb>, memref<1x!sycl_id_1_>) -> memref<?xf32, 1>
  // CHECK-DAG:     [[ACC1_END_PTR:%.*]] = "polygeist.memref2pointer"([[ACC1_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC2_BEGIN_PTR:%.*]]  = "polygeist.memref2pointer"([[ACC2_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %14 = llvm.icmp "ule" [[ACC1_END_PTR]], [[ACC2_BEGIN_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC1_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ACC1_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC2_END_PTR:%.*]] = "polygeist.memref2pointer"([[ACC2_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %17 = llvm.icmp "uge" [[ACC1_BEGIN_PTR]], [[ACC2_END_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %18 = arith.ori %14, %17 : i1
  // CHECK-NEXT:    scf.if %18 {
  // CHECK-NEXT:      func.call @callee1.specialized(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      func.call @callee1(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    }
  func.func private @callee1(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) {
    return
  }
  gpu.func @caller1(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) kernel {
    func.call @callee1(%arg0, %arg1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
    gpu.return
  }

  // COM: No need to version as the accessor types are different.
  // CHECK-LABEL: func.func private @callee2.specialized(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_1_i32_r_gb> {sycl.inner.disjoint}, 
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_1_f32_w_gb> {sycl.inner.disjoint}) attributes {llvm.linkage = #llvm.linkage<private>} {
  // CHECK-LABEL: func.func private @callee2(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_1_i32_r_gb>, 
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) {
  // CHECK-LABEL: gpu.func @caller2(%arg0: memref<?x!sycl_accessor_1_i32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) kernel {
  // CHECK-NEXT:    func.call @callee2.specialized(%arg0, %arg1) : (memref<?x!sycl_accessor_1_i32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    gpu.return
  // CHECK-NEXT:  }
  func.func private @callee2(%arg0: memref<?x!sycl_accessor_1_i32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) {
    return
  }
  gpu.func @caller2(%arg0: memref<?x!sycl_accessor_1_i32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) kernel {
    func.call @callee2(%arg0, %arg1) : (memref<?x!sycl_accessor_1_i32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
    gpu.return
  }

  /// COM: Check 2D accessors.
  // CHECK-LABEL: func.func private @callee3.specialized(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_2_f32_r_gb> {sycl.inner.disjoint}, 
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_2_f32_w_gb> {sycl.inner.disjoint})
  // CHECK-LABEL: func.func private @callee3(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_2_f32_r_gb>, 
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_2_f32_w_gb>)
  // CHECK-LABEL: gpu.func @caller3(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_accessor_2_f32_w_gb>) kernel {

  // COM: Obtain a pointer to the beginning of the first accessor.
  // CHECK-NEXT:    %alloca = memref.alloca() : memref<1x!sycl_id_2_>
  // CHECK-NEXT:    %c0 = arith.constant 0 : index
  // CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
  // CHECK-NEXT:    %0 = sycl.id.get %alloca[%c0_i32] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    memref.store %c0, %0[%c0] : memref<?xindex>
  // CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
  // CHECK-NEXT:    %1 = sycl.id.get %alloca[%c1_i32] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    memref.store %c0, %1[%c0] : memref<?xindex>
  // CHECK-NEXT:    [[ACC1_BEGIN:%.*]] = sycl.accessor.subscript %arg0[%alloca] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>

  // COM: Obtain a pointer to the end of the first accessor.
  // CHECK-NEXT:    %3 = sycl.accessor.get_range(%arg0) : (memref<?x!sycl_accessor_2_f32_r_gb>) -> !sycl_range_2_
  // CHECK-NEXT:    %alloca_0 = memref.alloca() : memref<1x!sycl_range_2_>
  // CHECK-NEXT:    %c0_1 = arith.constant 0 : index
  // CHECK-NEXT:    memref.store %3, %alloca_0[%c0_1] : memref<1x!sycl_range_2_>
  // CHECK-NEXT:    %alloca_2 = memref.alloca() : memref<1x!sycl_id_2_>
  // CHECK-NEXT:    %c1 = arith.constant 1 : index
  // CHECK-NEXT:    %c0_i32_3 = arith.constant 0 : i32
  // CHECK-NEXT:    %4 = sycl.id.get %alloca_2[%c0_i32_3] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    %c0_i32_4 = arith.constant 0 : i32
  // CHECK-NEXT:    %5 = sycl.range.get %alloca_0[%c0_i32_4] : (memref<1x!sycl_range_2_>, i32) -> index
  // CHECK-NEXT:    %6 = arith.subi %5, %c1 : index
  // CHECK-NEXT:    memref.store %6, %4[%c0_1] : memref<?xindex>
  // CHECK-NEXT:    %c1_i32_5 = arith.constant 1 : i32
  // CHECK-NEXT:    %7 = sycl.id.get %alloca_2[%c1_i32_5] : (memref<1x!sycl_id_2_>, i32) -> memref<?xindex>
  // CHECK-NEXT:    %c1_i32_6 = arith.constant 1 : i32
  // CHECK-NEXT:    %8 = sycl.range.get %alloca_0[%c1_i32_6] : (memref<1x!sycl_range_2_>, i32) -> index
  // CHECK-NEXT:    memref.store %8, %7[%c0_1] : memref<?xindex>
  // CHECK-NEXT:    [[ACC1_END:%.*]] = sycl.accessor.subscript %arg0[%alloca_2] : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>

  // COM: Version with condition: [[ACC1_END]] <= [[ACC2_BEGIN]] || [[ACC1_BEGIN]] >= [[ACC2_END]].
  // CHECK:         [[ACC2_BEGIN:%.*]] = sycl.accessor.subscript %arg1[%alloca_7] : (memref<?x!sycl_accessor_2_f32_w_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
  // CHECK:         [[ACC2_END:%.*]] = sycl.accessor.subscript %arg1[%alloca_13] : (memref<?x!sycl_accessor_2_f32_w_gb>, memref<1x!sycl_id_2_>) -> memref<?xf32, 1>
  // CHECK-DAG:     [[ACC1_END_PTR:%.*]] = "polygeist.memref2pointer"([[ACC1_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC2_BEGIN_PTR:%.*]]  = "polygeist.memref2pointer"([[ACC2_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %22 = llvm.icmp "ule" [[ACC1_END_PTR]], [[ACC2_BEGIN_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC1_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ACC1_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ACC2_END_PTR:%.*]] = "polygeist.memref2pointer"([[ACC2_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %25 = llvm.icmp "uge" [[ACC1_BEGIN_PTR]], [[ACC2_END_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %26 = arith.ori %22, %25 : i1
  // CHECK-NEXT:    scf.if %26 {
  // CHECK-NEXT:      func.call @callee3.specialized(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_accessor_2_f32_w_gb>) -> ()
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      func.call @callee3(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_accessor_2_f32_w_gb>) -> ()
  // CHECK-NEXT:    }
  func.func private @callee3(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_accessor_2_f32_w_gb>) {
    return
  }
  gpu.func @caller3(%arg0: memref<?x!sycl_accessor_2_f32_r_gb>, %arg1: memref<?x!sycl_accessor_2_f32_w_gb>) kernel {
    func.call @callee3(%arg0, %arg1) : (memref<?x!sycl_accessor_2_f32_r_gb>, memref<?x!sycl_accessor_2_f32_w_gb>) -> ()
    gpu.return
  }

  // COM: Check callee (@callee4) called indirectly from GPU kernel (@wrapper4).
  // CHECK-LABEL: func.func private @callee4.specialized(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_1_f32_r_gb> {sycl.inner.disjoint},
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_1_f32_w_gb> {sycl.inner.disjoint})
  // CHECK-LABEL: func.func private @callee4(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_1_f32_r_gb>,
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_1_f32_w_gb>)
  // CHECK-LABEL: func.func @caller4(%arg0: memref<?x!llvm.struct<(!sycl_accessor_1_f32_r_gb, !sycl_accessor_1_f32_w_gb)>>) {
  // CHECK:         scf.if %{{.*}} {
  // CHECK-NEXT:      func.call @callee4.specialized(%0, %1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      func.call @callee4(%0, %1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
  // CHECK-NEXT:    }
  // CHECK-LABEL: gpu.func @wrapper4(%arg0: memref<?x!llvm.struct<(!sycl_accessor_1_f32_r_gb, !sycl_accessor_1_f32_w_gb)>>) kernel {
  // CHECK-NEXT:    sycl.call @caller4(%arg0)
  func.func private @callee4(%arg0: memref<?x!sycl_accessor_1_f32_r_gb>, %arg1: memref<?x!sycl_accessor_1_f32_w_gb>) {
    return
  }
  func.func @caller4(%arg0: memref<?x!llvm.struct<(!sycl_accessor_1_f32_r_gb, !sycl_accessor_1_f32_w_gb)>>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %0 = "polygeist.subindex"(%arg0, %c0) : (memref<?x!llvm.struct<(!sycl_accessor_1_f32_r_gb, !sycl_accessor_1_f32_w_gb)>>, index) -> memref<?x!sycl_accessor_1_f32_r_gb>
    %1 = "polygeist.subindex"(%arg0, %c1) : (memref<?x!llvm.struct<(!sycl_accessor_1_f32_r_gb, !sycl_accessor_1_f32_w_gb)>>, index) -> memref<?x!sycl_accessor_1_f32_w_gb>
    func.call @callee4(%0, %1) : (memref<?x!sycl_accessor_1_f32_r_gb>, memref<?x!sycl_accessor_1_f32_w_gb>) -> ()
    return
  }
  gpu.func @wrapper4(%arg0: memref<?x!llvm.struct<(!sycl_accessor_1_f32_r_gb, !sycl_accessor_1_f32_w_gb)>>) kernel {
    sycl.call @caller4(%arg0) {MangledFunctionName = @caller4, TypeName = @RoundedRangeKernel}: (memref<?x!llvm.struct<(!sycl_accessor_1_f32_r_gb, !sycl_accessor_1_f32_w_gb)>>) -> ()
    gpu.return
  }

  // COM: Check accessors with dimension 0.
  // CHECK-LABEL: func.func private @callee5.specialized(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_0_f32_r_gb> {sycl.inner.disjoint},
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_0_f32_w_gb> {sycl.inner.disjoint})
  // CHECK-LABEL: func.func private @callee5(
  // CHECK-SAME:    %arg0: memref<?x!sycl_accessor_0_f32_r_gb>,
  // CHECK-SAME:    %arg1: memref<?x!sycl_accessor_0_f32_w_gb>)
  // CHECK-LABEL: gpu.func @caller5(%arg0: memref<?x!sycl_accessor_0_f32_r_gb>, %arg1: memref<?x!sycl_accessor_0_f32_w_gb>) kernel {
  // CHECK-NEXT:    [[ARG0_BEGIN:%.*]] = sycl.accessor.get_pointer(%arg0) : (memref<?x!sycl_accessor_0_f32_r_gb>) -> memref<?xf32, 1>
  // CHECK-NEXT:    %1 = sycl.accessor.get_pointer(%arg0) : (memref<?x!sycl_accessor_0_f32_r_gb>) -> memref<?xf32, 1>
  // CHECK-NEXT:    %c1 = arith.constant 1 : index
  // CHECK-NEXT:    [[ARG0_END:%.*]] = "polygeist.subindex"(%1, %c1) : (memref<?xf32, 1>, index) -> memref<?xf32, 1>
  // CHECK-NEXT:    [[ARG1_BEGIN:%.*]] = sycl.accessor.get_pointer(%arg1) : (memref<?x!sycl_accessor_0_f32_w_gb>) -> memref<?xf32, 1>
  // CHECK-NEXT:    %4 = sycl.accessor.get_pointer(%arg1) : (memref<?x!sycl_accessor_0_f32_w_gb>) -> memref<?xf32, 1>
  // CHECK-NEXT:    %c1_0 = arith.constant 1 : index
  // CHECK-NEXT:    [[ARG1_END:%.*]] = "polygeist.subindex"(%4, %c1_0) : (memref<?xf32, 1>, index) -> memref<?xf32, 1>
  // CHECK-DAG:     [[ARG0_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ARG1_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %8 = llvm.icmp "ule" [[ARG0_END_PTR]], [[ARG1_BEGIN_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ARG0_BEGIN_PTR:%.*]] = "polygeist.memref2pointer"([[ARG0_BEGIN]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-DAG:     [[ARG1_END_PTR:%.*]] = "polygeist.memref2pointer"([[ARG1_END]]) : (memref<?xf32, 1>) -> !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %11 = llvm.icmp "uge" [[ARG0_BEGIN_PTR]], [[ARG1_END_PTR]] : !llvm.ptr<f32, 1>
  // CHECK-NEXT:    %12 = arith.ori %8, %11 : i1
  // CHECK-NEXT:    scf.if %12 {
  // CHECK-NEXT:      func.call @callee5.specialized(%arg0, %arg1) : (memref<?x!sycl_accessor_0_f32_r_gb>, memref<?x!sycl_accessor_0_f32_w_gb>) -> ()
  // CHECK-NEXT:    } else {
  // CHECK-NEXT:      func.call @callee5(%arg0, %arg1) : (memref<?x!sycl_accessor_0_f32_r_gb>, memref<?x!sycl_accessor_0_f32_w_gb>) -> ()
  // CHECK-NEXT:    }
  func.func private @callee5(%arg0: memref<?x!sycl_accessor_0_f32_r_gb>, %arg1: memref<?x!sycl_accessor_0_f32_w_gb>) {
    return
  }
  gpu.func @caller5(%arg0: memref<?x!sycl_accessor_0_f32_r_gb>, %arg1: memref<?x!sycl_accessor_0_f32_w_gb>) kernel {
    func.call @callee5(%arg0, %arg1) : (memref<?x!sycl_accessor_0_f32_r_gb>, memref<?x!sycl_accessor_0_f32_w_gb>) -> ()
    gpu.return
  }
}
