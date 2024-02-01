// RUN: mlir-opt -split-input-file -verify-diagnostics %s

func.func @genx.fptofp(%a : i32) {
  // expected-error @+1 {{custom op 'genx.conv.fptofp' invalid kind of type specified}}
  %0 = genx.conv.fptofp %a {roundingMode=#genx.rounding_mode<RTE>} : i32 to i16
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op expecting repeat count to be 1, 2, 4, or 8}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=6:i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op expecting precision of matrix A and B to be the same}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<U8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi8>, %a : vector<16xi8>, %b : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op 1st operand (C) and result (D) should have the same type}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi8>, vector<16xi8>, vector<32xi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xi8>, %b : vector<8xi32>) {
  // expected-error @+1 {{'genx.matrix.dpas' op element type of 2nd (A) and 3rd (B) operands must match}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<S8>, pb=#genx.precision_type<S8>, rc=8:i32} : (vector<8xi32>, vector<16xi8>, vector<8xi32>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @genx.dpas(%c : vector<8xi32>, %a : vector<16xsi8>, %b : vector<32xsi8>) {
  // expected-error @+1 {{'genx.matrix.dpas' op precision should be S8 when 2nd (A) or 3rd (B) operand element type is signed i8}}
  %0 = genx.matrix.dpas %c, %a, %b {pa=#genx.precision_type<U8>, pb=#genx.precision_type<U8>, rc=8:i32} : (vector<8xi32>, vector<16xsi8>, vector<32xsi8>) -> vector<8xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=64:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op transpose and vnni transform are mutually exclusive}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=true, vnni_transform=true} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'genx.matrix.2Dblockload' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op tile_width for 32 bit elements should be equal to systolic depth, i.e., 8 elements}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=5:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<5xf32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op tile_width for 16 bit elements should be equal to systolic depth times 2, i.e., 16 elements}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xf16>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op tile_width for 8 bit elements should be equal to systolic depth times 4, i.e., 32 elements}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<4xi8>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op tile_height for 32 bit elements should be 8}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=32:i32, tile_width=8:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<8xf32>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op tile_height for 16 bit elements should be 16}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=16:i32, tile_width=16:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<16xf16>
  llvm.return
}

// -----

func.func @matrix_2Dblockload(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32) {
  // expected-error @+1 {{'genx.matrix.2Dblockload' op tile_height for 8 bit elements should be 32}}
  %0 = genx.matrix.2Dblockload %ptr, %base_width, %base_height, %base_pitch, %x, %y {elem_size_in_bits=8:i32, tile_width=32:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32) -> vector<32xi8>
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op expecting 'elem_size_in_bits' to be 8, 16, or 32}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=64:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op transpose and vnni transform are mutually exclusive}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=true, vnni_transform=true} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_height : i32, %x : i32, %y : i32, %stored_val : vector<4xi32>) {
  %base_width = llvm.mlir.constant(4 : i32) : i32
  %base_pitch = llvm.mlir.constant(2 : i32) : i32
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op 4th operand (base pitch) should be >= 2nd operand (base width)}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi32>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xf32>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op tile_width for 32 bit elements should be equal to systolic depth, i.e., 8 elements}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xf32>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xf16>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op tile_width for 16 bit elements should be equal to systolic depth times 2, i.e., 16 elements}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=16:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xf16>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<4xi8>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op tile_width for 8 bit elements should be equal to systolic depth times 4, i.e., 32 elements}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8:i32, tile_width=4:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<4xi8>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<8xf32>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op tile_height for 32 bit elements should be 8}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=32:i32, tile_width=8:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<8xf32>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<16xf16>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op tile_height for 16 bit elements should be 16}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=16:i32, tile_width=16:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<16xf16>)
  llvm.return
}

// -----

func.func @matrix_2Dblockstore(%ptr : !llvm.ptr, %base_width : i32, %base_height : i32, %base_pitch : i32, %x : i32, %y : i32, %stored_val : vector<32xi8>) {
  // expected-error @+1 {{'genx.matrix.2Dblockstore' op tile_height for 8 bit elements should be 32}}
  genx.matrix.2Dblockstore %ptr, %base_width, %base_height, %base_pitch, %x, %y, %stored_val {elem_size_in_bits=8:i32, tile_width=32:i32, tile_height=1:i32, v_blocks=1:i32, transpose=false, vnni_transform=false} : (!llvm.ptr, i32, i32, i32, i32, i32, vector<32xi8>)
  llvm.return
}

