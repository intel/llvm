// RUN: mlir-translate -mlir-to-llvmir -split-input-file %s | FileCheck %s

llvm.func @genx_special_regs() -> i64 {
  // CHECK-LABEL: genx_special_regs
  // CHECK: call i64 @_Z12get_local_idj(i32 0)
  %1 = genx.workitem.id.x : i64
  // CHECK: call i64 @_Z12get_local_idj(i32 1)
  %2 = genx.workitem.id.y : i64
  // CHECK: call i64 @_Z12get_local_idj(i32 2)
  %3 = genx.workitem.id.z : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 0)
  %4 = genx.workgroup.id.x : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 1)
  %5 = genx.workgroup.id.y : i64
  // CHECK: call i64 @_Z12get_group_idj(i32 2)
  %6 = genx.workgroup.id.z : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 0)
  %7 = genx.workgroup.dim.x : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 1)
  %8 = genx.workgroup.dim.y : i64
  // CHECK: call i64 @_Z12get_local_sizej(i32 2)
  %9 = genx.workgroup.dim.z : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 0)
  %10 = genx.grid.dim.x : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 1)
  %11 = genx.grid.dim.y : i64
  // CHECK: call i64 @_Z12get_global_sizej(i32 2)
  %12 = genx.grid.dim.z : i64

  llvm.return %1 : i64
}

// -----

llvm.func @genx.barrier() {
  // CHECK-LABEL: genx.barrier
  // CHECK: call void @_Z7barrierj(i32 3) [[ATTR:#.*]]
  genx.barrier
  llvm.return
}
// CHECK: attributes [[ATTR]] = { convergent }

// -----


llvm.func @genx.sub_group_shuffle() {
  // CHECK-LABEL: genx.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = call i32 @_Z21sub_group_shuffle_xorij(i32 0, i32 0)
  %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  // CHECK: %2 = call i32 @_Z20sub_group_shuffle_upij(i32 0, i32 0)
  %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  // CHECK: %3 = call i32 @_Z22sub_group_shuffle_downij(i32 0, i32 0)
  %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  // CHECK: %4 = call i32 @_Z17sub_group_shuffleij(i32 0, i32 0)
  %4 = genx.sub_group_shuffle IDX %0, %0 : i32 -> i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %5 = call i8 @_Z21sub_group_shuffle_xorcj(i8 0, i32 0)
  %6 = genx.sub_group_shuffle XOR %5, %0 : i8 -> i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %6 = call i16 @_Z21sub_group_shuffle_xorsj(i16 0, i32 0)
  %8 = genx.sub_group_shuffle XOR %7, %0 : i16 -> i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %7 = call i64 @_Z21sub_group_shuffle_xorlj(i64 0, i32 0)
  %10 = genx.sub_group_shuffle XOR %9, %0 : i64 -> i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %8 = call half @_Z21sub_group_shuffle_xorDhj(half 0xH0000, i32 0)
  %12 = genx.sub_group_shuffle XOR %11, %0 : f16 -> f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %9 = call float @_Z21sub_group_shuffle_xorfj(float 0.000000e+00, i32 0)
  %14 = genx.sub_group_shuffle XOR %13, %0 : f32 -> f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %10 = call double @_Z21sub_group_shuffle_xordj(double 0.000000e+00, i32 0)
  %16 = genx.sub_group_shuffle XOR %15, %0 : f64 -> f64
  llvm.return
}

// -----

llvm.func @genx.dpas.f32(%c : vector<8xf32>, %a : vector<4xf32>, %b : vector<8xf32>) {
  // CHECK-DAG:  [[A:%.*]] = bitcast <4 x float> %1 to <8 x i16>
  // CHECK-DAG:  [[B:%.*]] = bitcast <8 x float> %2 to <8 x i32>
  // CHECK-NEXT: call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %0, <8 x i16> [[A]], <8 x i32> [[B]], i32 8, i32 8, i32 8, i32 8, i1 false)
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<TF32>, pb=#genx.precision_type<TF32>, rc=8:i32} : (vector<8xf32>, vector<4xf32>, vector<8xf32>) -> vector<8xf32>
  llvm.return
}

llvm.func @genx.dpas.f16(%c : vector<8xf32>, %a : vector<8xf16>, %b : vector<16xf16>) {
  // CHECK-DAG:  [[A:%.*]] = bitcast <8 x half> %1 to <8 x i16>
  // CHECK-DAG:  [[B:%.*]] = bitcast <16 x half> %2 to <8 x i32>
  // CHECK-NEXT: call <8 x float> @llvm.genx.GenISA.sub.group.dpas.v8f32.v8f32.v8i16.v8i32(<8 x float> %0, <8 x i16> [[A]], <8 x i32> [[B]], i32 10, i32 10, i32 8, i32 8, i1 false)
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<FP16>, pb=#genx.precision_type<FP16>, rc=8:i32} : (vector<8xf32>, vector<8xf16>, vector<16xf16>) -> vector<8xf32>
  llvm.return
}

llvm.func @genx.dpas.i8(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // CHECK-DAG:  [[A:%.*]] = bitcast <16 x i8> %1 to <8 x i16>
  // CHECK-DAG:  [[B:%.*]] = bitcast <32 x i8> %2 to <8 x i32>
  // CHECK-NEXT: call <8 x i32> @llvm.genx.GenISA.sub.group.dpas.v8i32.v8i32.v8i16.v8i32(<8 x i32> %0, <8 x i16> [[A]], <8 x i32> [[B]], i32 4, i32 4, i32 8, i32 8, i1 false)
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

llvm.func @genx.2Dblockload(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: [[PTR:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-NEXT: call <4 x i32> @llvm.genx.GenISA.LSC2DBlockRead.v4i32(i64 [[PTR]], i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 32, i32 4, i32 1, i32 1, i1 false, i1 false)
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

llvm.func @genx.2Dblockstore(%ptr : !llvm.ptr<i32>, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // CHECK: [[PTR:%.*]] = ptrtoint ptr %0 to i64
  // CHECK-NEXT: call void @llvm.genx.GenISA.LSC2DBlockWrite.v4i32(i64 [[PTR]], i32 %1, i32 %2, i32 %3, i32 %4, i32 %5, i32 32, i32 4, i32 1, i32 1, i1 false, i1 false, <4 x i32> %6)
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr<i32>, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}
