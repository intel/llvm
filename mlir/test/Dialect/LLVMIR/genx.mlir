// RUN: mlir-opt %s -split-input-file -verify-diagnostics | FileCheck %s

func.func @genx_special_regs() -> i32 {
  // CHECK-LABEL: genx_special_regs
  // CHECK: genx.workitem.id.x : i32
  %0 = genx.workitem.id.x : i32
  // CHECK: genx.workitem.id.y : i32
  %1 = genx.workitem.id.y : i32
  // CHECK: genx.workitem.id.z : i32
  %2 = genx.workitem.id.z : i32
  // CHECK: genx.workgroup.id.x : i32
  %3 = genx.workgroup.id.x : i32
  // CHECK: genx.workgroup.id.y : i32
  %4 = genx.workgroup.id.y : i32
  // CHECK: genx.workgroup.id.z : i32
  %5 = genx.workgroup.id.z : i32
  // CHECK: genx.workgroup.dim.x : i32
  %6 = genx.workgroup.dim.x : i32
  // CHECK: genx.workgroup.dim.y : i32
  %7 = genx.workgroup.dim.y : i32
  // CHECK: genx.workgroup.dim.z : i32
  %8 = genx.workgroup.dim.z : i32
  // CHECK: genx.grid.dim.x : i32
  %9 = genx.grid.dim.x : i32
  // CHECK: genx.grid.dim.y : i32
  %10 = genx.grid.dim.y : i32
  // CHECK: genx.grid.dim.z : i32
  %11 = genx.grid.dim.z : i32
  llvm.return %0 : i32
}

func.func @genx.barrier() {
  // CHECK-LABEL: genx.barrier
  // CHECK: genx.barrier
  genx.barrier
  llvm.return
}

func.func @genx.sub_group_shuffle() {
  // CHECK-LABEL: genx.sub_group_shuffle
  %0 = llvm.mlir.constant(0 : i32) : i32
  // CHECK: %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  %1 = genx.sub_group_shuffle XOR %0, %0 : i32 -> i32
  // CHECK: %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  %2 = genx.sub_group_shuffle UP %0, %0 : i32 -> i32
  // CHECK: %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  %3 = genx.sub_group_shuffle DOWN %0, %0 : i32 -> i32
  // CHECK: %4 = genx.sub_group_shuffle IDX %0, %0 : i32 -> i32
  %4 = genx.sub_group_shuffle IDX %0, %0 : i32 -> i32
  %5 = llvm.mlir.constant(0 : i8) : i8
  // CHECK: %6 = genx.sub_group_shuffle XOR %5, %0 : i8 -> i8
  %6 = genx.sub_group_shuffle XOR %5, %0 : i8 -> i8
  %7 = llvm.mlir.constant(0 : i16) : i16
  // CHECK: %8 = genx.sub_group_shuffle XOR %7, %0 : i16 -> i16
  %8 = genx.sub_group_shuffle XOR %7, %0 : i16 -> i16
  %9 = llvm.mlir.constant(0 : i64) : i64
  // CHECK: %10 = genx.sub_group_shuffle XOR %9, %0 : i64 -> i64
  %10 = genx.sub_group_shuffle XOR %9, %0 : i64 -> i64
  %11 = llvm.mlir.constant(0.0 : f16) : f16
  // CHECK: %12 = genx.sub_group_shuffle XOR %11, %0 : f16 -> f16
  %12 = genx.sub_group_shuffle XOR %11, %0 : f16 -> f16
  %13 = llvm.mlir.constant(0.0 : f32) : f32
  // CHECK: %14 = genx.sub_group_shuffle XOR %13, %0 : f32 -> f32
  %14 = genx.sub_group_shuffle XOR %13, %0 : f32 -> f32
  %15 = llvm.mlir.constant(0.0 : f64) : f64
  // CHECK: %16 = genx.sub_group_shuffle XOR %15, %0 : f64 -> f64
  %16 = genx.sub_group_shuffle XOR %15, %0 : f64 -> f64
  llvm.return
}

llvm.func @genx.fptofp(%a: f32, %b: f16) {
  // CHECK: %0 = genx.conv.fptofp %arg0 {roundingMode = #genx.rounding_mode<RTE>} : f32 to f16
  // CHECK-NEXT: %1 = genx.conv.fptofp %arg0 {roundingMode = #genx.rounding_mode<RTN>} : f32 to f16
  // CHECK-NEXT: %2 = genx.conv.fptofp %arg0 {roundingMode = #genx.rounding_mode<RTP>} : f32 to f16
  // CHECK-NEXT: %3 = genx.conv.fptofp %arg0 {roundingMode = #genx.rounding_mode<RTZ>} : f32 to f16
  // CHECK-NEXT: %4 = genx.conv.fptofp %arg1 {roundingMode = #genx.rounding_mode<RTE>} : f16 to f32
  // CHECK-NEXT: %5 = genx.conv.fptofp %arg1 {roundingMode = #genx.rounding_mode<RTN>} : f16 to f32
  // CHECK-NEXT: %6 = genx.conv.fptofp %arg1 {roundingMode = #genx.rounding_mode<RTP>} : f16 to f32
  // CHECK-NEXT: %7 = genx.conv.fptofp %arg1 {roundingMode = #genx.rounding_mode<RTZ>} : f16 to f32
  %0 = genx.conv.fptofp %a {roundingMode=#genx.rounding_mode<RTE>} : f32 to f16
  %1 = genx.conv.fptofp %a {roundingMode=#genx.rounding_mode<RTN>} : f32 to f16
  %2 = genx.conv.fptofp %a {roundingMode=#genx.rounding_mode<RTP>} : f32 to f16
  %3 = genx.conv.fptofp %a {roundingMode=#genx.rounding_mode<RTZ>} : f32 to f16
  %4 = genx.conv.fptofp %b {roundingMode=#genx.rounding_mode<RTE>} : f16 to f32
  %5 = genx.conv.fptofp %b {roundingMode=#genx.rounding_mode<RTN>} : f16 to f32
  %6 = genx.conv.fptofp %b {roundingMode=#genx.rounding_mode<RTP>} : f16 to f32
  %7 = genx.conv.fptofp %b {roundingMode=#genx.rounding_mode<RTZ>} : f16 to f32
  llvm.return
}

llvm.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // CHECK: %0 = genx.matrix.dpas %arg0, %arg1, %arg2 {pa = #genx.precision_type<S8>, pb = #genx.precision_type<S8>, rc = 8 : i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

func.func @genx.2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // CHECK: %0 = genx.matrix.2Dblockload %arg0, %arg1, %arg2, %arg3, %arg4, %arg5 {elem_size_in_bits = 16 : i32, tile_height = 16 : i32, tile_width = 16 : i32, transpose = false, v_blocks = 1 : i32, vnni_transform = false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16:i32, tile_width=16:i32, tile_height=16:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  llvm.return
}

func.func @genx.2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xf32>) {
  // CHECK: genx.matrix.2Dblockstore %arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6 {elem_size_in_bits = 32 : i32, tile_height = 8 : i32, tile_width = 8 : i32, transpose = false, v_blocks = 1 : i32, vnni_transform = false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=8:i32, tile_height=8:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  llvm.return
}
