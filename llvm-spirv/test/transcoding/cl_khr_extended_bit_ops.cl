// RUN: %clang_cc1 -triple spir-unknown-unknown -O1 -cl-std=CL2.0 -fdeclare-opencl-builtins -finclude-default-header -emit-llvm-bc %s -o %t.bc
// RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_bit_instructions -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
// RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_bit_instructions -o %t.spv
// RUN: llvm-spirv -r %t.spv -o %t.rev.bc
// RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

// TODO: remove prototypes once bit op builtins are available in clang.
#define __ovld __attribute__((overloadable))

int2 __ovld bitfield_insert(int2, int2, uint, uint);
short __ovld bitfield_extract_signed(short, uint, uint);
short __ovld bitfield_extract_signed(ushort, uint, uint);
uchar8 __ovld bitfield_extract_unsigned(char8, uint, uint);
uchar8 __ovld bitfield_extract_unsigned(uchar8, uint, uint);
ulong4 __ovld bit_reverse(ulong4);

// CHECK-SPIRV: Capability BitInstructions
// CHECK-SPIRV: Extension "SPV_KHR_bit_instructions"

// CHECK-LLVM-LABEL: @testInsert
// CHECK-LLVM: call spir_func <2 x i32> @_Z15bitfield_insertDv2_iS_jj
// CHECK-SPIRV: Function
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[insbase:[0-9]+]]
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[insins:[0-9]+]]
// CHECK-SPIRV: BitFieldInsert {{[0-9]+}} {{[0-9]+}} [[insbase]] [[insins]]
kernel void testInsert(int2 b, int2 i, global int2 *res) {
  *res = bitfield_insert(b, i, 4, 2);
}

// CHECK-LLVM-LABEL: @testExtractS
// CHECK-LLVM: call spir_func i16 @_Z23bitfield_extract_signedsjj
// CHECK-LLVM: call spir_func i16 @_Z23bitfield_extract_signedsjj
// CHECK-SPIRV: Function
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[sextrbase:[0-9]+]]
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[sextrbaseu:[0-9]+]]
// CHECK-SPIRV: BitFieldSExtract {{[0-9]+}} {{[0-9]+}} [[sextrbase]]
// CHECK-SPIRV: BitFieldSExtract {{[0-9]+}} {{[0-9]+}} [[sextrbaseu]]
kernel void testExtractS(short b, ushort bu, global short *res) {
  *res = bitfield_extract_signed(b, 5, 4);
  *res += bitfield_extract_signed(bu, 5, 4);
}

// CHECK-LLVM-LABEL: @testExtractU
// CHECK-LLVM: call spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_cjj
// CHECK-LLVM: call spir_func <8 x i8> @_Z25bitfield_extract_unsignedDv8_cjj
// CHECK-SPIRV: Function
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[uextrbase:[0-9]+]]
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[uextrbaseu:[0-9]+]]
// CHECK-SPIRV: BitFieldUExtract {{[0-9]+}} {{[0-9]+}} [[uextrbase]]
// CHECK-SPIRV: BitFieldUExtract {{[0-9]+}} {{[0-9]+}} [[uextrbaseu]]
kernel void testExtractU(char8 b, uchar8 bu, global uchar8 *res) {
  *res = bitfield_extract_unsigned(b, 3, 4);
  *res += bitfield_extract_unsigned(bu, 3, 4);
}

// CHECK-LLVM-LABEL: @testBitReverse
// CHECK-LLVM: call <4 x i64> @llvm.bitreverse.v4i64
// CHECK-SPIRV: Function
// CHECK-SPIRV: FunctionParameter {{[0-9]+}} [[revbase:[0-9]+]]
// CHECK-SPIRV: BitReverse {{[0-9]+}} {{[0-9]+}} [[revbase]]
kernel void testBitReverse(ulong4 b, global ulong4 *res) {
  *res = bit_reverse(b);
}
