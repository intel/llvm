; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_subgroup_bitcast_shuffle -o %t.spv
; RUN: llvm-spirv %t.spv --to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 -o %t.err.spv | FileCheck %s --check-prefix=CHECK-ERROR
; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_subgroup_bitcast_shuffle

; CHECK-SPIRV-DAG: Capability SubgroupBitcastShuffleINTEL
; CHECK-SPIRV-DAG: Extension "SPV_INTEL_subgroup_bitcast_shuffle"

; CHECK-SPIRV-DAG: TypeInt [[#TI8:]] 8 0
; CHECK-SPIRV-DAG: TypeInt [[#TI16:]] 16 0
; CHECK-SPIRV-DAG: TypeInt [[#TI32:]] 32 0
; CHECK-SPIRV-DAG: TypeInt [[#TI64:]] 64 0
; CHECK-SPIRV-DAG: TypeVector [[#TV2I8:]] [[#TI8]] 2
; CHECK-SPIRV-DAG: TypeVector [[#TV4I8:]] [[#TI8]] 4
; CHECK-SPIRV-DAG: TypeVector [[#TV8I8:]] [[#TI8]] 8
; CHECK-SPIRV-DAG: TypeVector [[#TV2I16:]] [[#TI16]] 2
; CHECK-SPIRV-DAG: TypeVector [[#TV4I16:]] [[#TI16]] 4
; CHECK-SPIRV-DAG: TypeVector [[#TV2I32:]] [[#TI32]] 2
; CHECK-SPIRV-DAG: TypeVector [[#TV16I8:]] [[#TI8]] 16
; CHECK-SPIRV-DAG: TypeVector [[#TV8I16:]] [[#TI16]] 8
; CHECK-SPIRV-DAG: TypeVector [[#TV4I32:]] [[#TI32]] 4
; CHECK-SPIRV-DAG: TypeVector [[#TV2I64:]] [[#TI64]] 2
; CHECK-SPIRV-DAG: TypeVector [[#TV16I16:]] [[#TI16]] 16
; CHECK-SPIRV-DAG: TypeVector [[#TV8I32:]] [[#TI32]] 8
; CHECK-SPIRV-DAG: TypeVector [[#TV4I64:]] [[#TI64]] 4
; CHECK-SPIRV-DAG: TypeVector [[#TV16I32:]] [[#TI32]] 16
; CHECK-SPIRV-DAG: TypeVector [[#TV8I64:]] [[#TI64]] 8

; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TI16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TI32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TI32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TI64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TI64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TI64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV16I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV16I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV16I8]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV2I64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV16I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV16I16]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV4I64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I32]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV8I64]]
; CHECK-SPIRV: SubgroupBitcastShuffleINTEL [[#TV16I32]]

; CHECK-LLVM-LABEL: define spir_kernel void @test_16bit()
; CHECK-LLVM: call spir_func i16 @_Z43__spirv_SubgroupBitcastShuffleINTEL_RushortDv2_c(<2 x i8>
; CHECK-LLVM: call spir_func <2 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar2s(i16

; CHECK-LLVM-LABEL: define spir_kernel void @test_32bit()
; CHECK-LLVM: call spir_func <2 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort2Dv4_c(<4 x i8>
; CHECK-LLVM: call spir_func <4 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar4Dv2_s(<2 x i16>
; CHECK-LLVM: call spir_func i32 @_Z41__spirv_SubgroupBitcastShuffleINTEL_RuintDv4_c(<4 x i8>
; CHECK-LLVM: call spir_func <4 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar4i(i32
; CHECK-LLVM: call spir_func i32 @_Z41__spirv_SubgroupBitcastShuffleINTEL_RuintDv2_s(<2 x i16>
; CHECK-LLVM: call spir_func <2 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort2i(i32

; CHECK-LLVM-LABEL: define spir_kernel void @test_64bit()
; CHECK-LLVM: call spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4Dv8_c(<8 x i8>
; CHECK-LLVM: call spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8Dv4_s(<4 x i16>
; CHECK-LLVM: call spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2Dv8_c(<8 x i8>
; CHECK-LLVM: call spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8Dv2_i(<2 x i32>
; CHECK-LLVM: call spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv8_c(<8 x i8>
; CHECK-LLVM: call spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8l(i64
; CHECK-LLVM: call spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2Dv4_s(<4 x i16>
; CHECK-LLVM: call spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4Dv2_i(<2 x i32>
; CHECK-LLVM: call spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv4_s(<4 x i16>
; CHECK-LLVM: call spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4l(i64
; CHECK-LLVM: call spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv2_i(<2 x i32>
; CHECK-LLVM: call spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2l(i64

; CHECK-LLVM-LABEL: define spir_kernel void @test_128bit()
; CHECK-LLVM: call spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv16_c(<16 x i8>
; CHECK-LLVM: call spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv8_s(<8 x i16>
; CHECK-LLVM: call spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv16_c(<16 x i8>
; CHECK-LLVM: call spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv4_i(<4 x i32>
; CHECK-LLVM: call spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv16_c(<16 x i8>
; CHECK-LLVM: call spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv2_l(<2 x i64>
; CHECK-LLVM: call spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv8_s(<8 x i16>
; CHECK-LLVM: call spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv4_i(<4 x i32>
; CHECK-LLVM: call spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv8_s(<8 x i16>
; CHECK-LLVM: call spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv2_l(<2 x i64>
; CHECK-LLVM: call spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv4_i(<4 x i32>
; CHECK-LLVM: call spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv2_l(<2 x i64>

; CHECK-LLVM-LABEL: define spir_kernel void @test_256bit()
; CHECK-LLVM: call spir_func <8 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint8Dv16_s(<16 x i16>
; CHECK-LLVM: call spir_func <16 x i16> @_Z45__spirv_SubgroupBitcastShuffleINTEL_Rushort16Dv8_i(<8 x i32>
; CHECK-LLVM: call spir_func <4 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong4Dv16_s(<16 x i16>
; CHECK-LLVM: call spir_func <16 x i16> @_Z45__spirv_SubgroupBitcastShuffleINTEL_Rushort16Dv4_l(<4 x i64>
; CHECK-LLVM: call spir_func <4 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong4Dv8_i(<8 x i32>
; CHECK-LLVM: call spir_func <8 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint8Dv4_l(<4 x i64>

; CHECK-LLVM-LABEL: define spir_kernel void @test_512bit()
; CHECK-LLVM: call spir_func <8 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong8Dv16_i(<16 x i32>
; CHECK-LLVM: call spir_func <16 x i32> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruint16Dv8_l(<8 x i64>

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir"

define spir_kernel void @test_16bit() {
entry:
  %v2i8 = load <2 x i8>, ptr undef
  %s16 = load i16, ptr undef

  %r2x8to16 = call spir_func i16 @_Z43__spirv_SubgroupBitcastShuffleINTEL_RushortDv2_c(<2 x i8> %v2i8)
  %r16to2x8 = call spir_func <2 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar2s(i16 %s16)
  ret void
}

define spir_kernel void @test_32bit() {
entry:
  %v4i8 = load <4 x i8>, ptr undef
  %v2i16 = load <2 x i16>, ptr undef
  %s32 = load i32, ptr undef

  %r4x8to2x16 = call spir_func <2 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort2Dv4_c(<4 x i8> %v4i8)
  %r2x16to4x8 = call spir_func <4 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar4Dv2_s(<2 x i16> %v2i16)
  %r4x8to32 = call spir_func i32 @_Z41__spirv_SubgroupBitcastShuffleINTEL_RuintDv4_c(<4 x i8> %v4i8)
  %r32to4x8 = call spir_func <4 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar4i(i32 %s32)
  %r2x16to32 = call spir_func i32 @_Z41__spirv_SubgroupBitcastShuffleINTEL_RuintDv2_s(<2 x i16> %v2i16)
  %r32to2x16 = call spir_func <2 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort2i(i32 %s32)
  ret void
}

define spir_kernel void @test_64bit() {
entry:
  %v8i8 = load <8 x i8>, ptr undef
  %v4i16 = load <4 x i16>, ptr undef
  %v2i32 = load <2 x i32>, ptr undef
  %s64 = load i64, ptr undef

  %r8x8to4x16 = call spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4Dv8_c(<8 x i8> %v8i8)
  %r4x16to8x8 = call spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8Dv4_s(<4 x i16> %v4i16)
  %r8x8to2x32 = call spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2Dv8_c(<8 x i8> %v8i8)
  %r2x32to8x8 = call spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8Dv2_i(<2 x i32> %v2i32)
  %r8x8to64 = call spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv8_c(<8 x i8> %v8i8)
  %r64to8x8 = call spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8l(i64 %s64)
  %r4x16to2x32 = call spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2Dv4_s(<4 x i16> %v4i16)
  %r2x32to4x16 = call spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4Dv2_i(<2 x i32> %v2i32)
  %r4x16to64 = call spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv4_s(<4 x i16> %v4i16)
  %r64to4x16 = call spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4l(i64 %s64)
  %r2x32to64 = call spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv2_i(<2 x i32> %v2i32)
  %r64to2x32 = call spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2l(i64 %s64)
  ret void
}

define spir_kernel void @test_128bit() {
entry:
  %v16i8 = load <16 x i8>, ptr undef
  %v8i16 = load <8 x i16>, ptr undef
  %v4i32 = load <4 x i32>, ptr undef
  %v2i64 = load <2 x i64>, ptr undef

  %r16x8to8x16 = call spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv16_c(<16 x i8> %v16i8)
  %r8x16to16x8 = call spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv8_s(<8 x i16> %v8i16)
  %r16x8to4x32 = call spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv16_c(<16 x i8> %v16i8)
  %r4x32to16x8 = call spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv4_i(<4 x i32> %v4i32)
  %r16x8to2x64 = call spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv16_c(<16 x i8> %v16i8)
  %r2x64to16x8 = call spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv2_l(<2 x i64> %v2i64)
  %r8x16to4x32 = call spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv8_s(<8 x i16> %v8i16)
  %r4x32to8x16 = call spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv4_i(<4 x i32> %v4i32)
  %r8x16to2x64 = call spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv8_s(<8 x i16> %v8i16)
  %r2x64to8x16 = call spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv2_l(<2 x i64> %v2i64)
  %r4x32to2x64 = call spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv4_i(<4 x i32> %v4i32)
  %r2x64to4x32 = call spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv2_l(<2 x i64> %v2i64)
  ret void
}

define spir_kernel void @test_256bit() {
entry:
  %v16i16 = load <16 x i16>, ptr undef
  %v8i32 = load <8 x i32>, ptr undef
  %v4i64 = load <4 x i64>, ptr undef

  %r16x16to8x32 = call spir_func <8 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint8Dv16_s(<16 x i16> %v16i16)
  %r8x32to16x16 = call spir_func <16 x i16> @_Z45__spirv_SubgroupBitcastShuffleINTEL_Rushort16Dv8_i(<8 x i32> %v8i32)
  %r16x16to4x64 = call spir_func <4 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong4Dv16_s(<16 x i16> %v16i16)
  %r4x64to16x16 = call spir_func <16 x i16> @_Z45__spirv_SubgroupBitcastShuffleINTEL_Rushort16Dv4_l(<4 x i64> %v4i64)
  %r8x32to4x64 = call spir_func <4 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong4Dv8_i(<8 x i32> %v8i32)
  %r4x64to8x32 = call spir_func <8 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint8Dv4_l(<4 x i64> %v4i64)
  ret void
}

define spir_kernel void @test_512bit() {
entry:
  %v16i32 = load <16 x i32>, ptr undef
  %v8i64 = load <8 x i64>, ptr undef

  %r16x32to8x64 = call spir_func <8 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong8Dv16_i(<16 x i32> %v16i32)
  %r8x64to16x32 = call spir_func <16 x i32> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruint16Dv8_l(<8 x i64> %v8i64)
  ret void
}

declare dso_local spir_func i16 @_Z43__spirv_SubgroupBitcastShuffleINTEL_RushortDv2_c(<2 x i8>)
declare dso_local spir_func <2 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar2s(i16)

declare dso_local spir_func <2 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort2Dv4_c(<4 x i8>)
declare dso_local spir_func <4 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar4Dv2_s(<2 x i16>)
declare dso_local spir_func i32 @_Z41__spirv_SubgroupBitcastShuffleINTEL_RuintDv4_c(<4 x i8>)
declare dso_local spir_func <4 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar4i(i32)
declare dso_local spir_func i32 @_Z41__spirv_SubgroupBitcastShuffleINTEL_RuintDv2_s(<2 x i16>)
declare dso_local spir_func <2 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort2i(i32)

declare dso_local spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4Dv8_c(<8 x i8>)
declare dso_local spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8Dv4_s(<4 x i16>)
declare dso_local spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2Dv8_c(<8 x i8>)
declare dso_local spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8Dv2_i(<2 x i32>)
declare dso_local spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv8_c(<8 x i8>)
declare dso_local spir_func <8 x i8> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruchar8l(i64)
declare dso_local spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2Dv4_s(<4 x i16>)
declare dso_local spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4Dv2_i(<2 x i32>)
declare dso_local spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv4_s(<4 x i16>)
declare dso_local spir_func <4 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort4l(i64)
declare dso_local spir_func i64 @_Z42__spirv_SubgroupBitcastShuffleINTEL_RulongDv2_i(<2 x i32>)
declare dso_local spir_func <2 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint2l(i64)

declare dso_local spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv16_c(<16 x i8>)
declare dso_local spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv8_s(<8 x i16>)
declare dso_local spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv16_c(<16 x i8>)
declare dso_local spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv4_i(<4 x i32>)
declare dso_local spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv16_c(<16 x i8>)
declare dso_local spir_func <16 x i8> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Ruchar16Dv2_l(<2 x i64>)
declare dso_local spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv8_s(<8 x i16>)
declare dso_local spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv4_i(<4 x i32>)
declare dso_local spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv8_s(<8 x i16>)
declare dso_local spir_func <8 x i16> @_Z44__spirv_SubgroupBitcastShuffleINTEL_Rushort8Dv2_l(<2 x i64>)
declare dso_local spir_func <2 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong2Dv4_i(<4 x i32>)
declare dso_local spir_func <4 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint4Dv2_l(<2 x i64>)

declare dso_local spir_func <8 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint8Dv16_s(<16 x i16>)
declare dso_local spir_func <16 x i16> @_Z45__spirv_SubgroupBitcastShuffleINTEL_Rushort16Dv8_i(<8 x i32>)
declare dso_local spir_func <4 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong4Dv16_s(<16 x i16>)
declare dso_local spir_func <16 x i16> @_Z45__spirv_SubgroupBitcastShuffleINTEL_Rushort16Dv4_l(<4 x i64>)
declare dso_local spir_func <4 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong4Dv8_i(<8 x i32>)
declare dso_local spir_func <8 x i32> @_Z42__spirv_SubgroupBitcastShuffleINTEL_Ruint8Dv4_l(<4 x i64>)

declare dso_local spir_func <8 x i64> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Rulong8Dv16_i(<16 x i32>)
declare dso_local spir_func <16 x i32> @_Z43__spirv_SubgroupBitcastShuffleINTEL_Ruint16Dv8_l(<8 x i64>)

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
