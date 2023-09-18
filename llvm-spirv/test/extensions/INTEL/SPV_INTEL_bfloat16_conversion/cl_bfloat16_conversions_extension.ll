; RUN: llvm-as %s -o %t.bc

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-WO-EXT

; RUN: llvm-spirv -s %t.bc -o %t.regularized.bc
; RUN: llvm-dis %t.regularized.bc -o %t.regularized.ll
; RUN: FileCheck < %t.regularized.ll %s --check-prefix=CHECK-REGULARIZED

; RUN: llvm-spirv --spirv-text %t.bc -o %t.spt --spirv-ext=+SPV_INTEL_bfloat16_conversion
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -to-binary %t.spt -o %t.spv

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=CL2.0
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM-CL20

; RUN: llvm-spirv -r %t.spv -o %t.rev.bc --spirv-target-env=SPV-IR
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM-SPV

; CHECK-WO-EXT: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-WO-EXT-NEXT: SPV_INTEL_bfloat16_conversion

; CHECK-REGULARIZED: call spir_func zeroext i16 @_Z27__spirv_ConvertFToBF16INTELf(float 0.000000e+00)
; CHECK-REGULARIZED: call spir_func <2 x i16> @_Z27__spirv_ConvertFToBF16INTELDv2_f(<2 x float> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <3 x i16> @_Z27__spirv_ConvertFToBF16INTELDv3_f(<3 x float> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <4 x i16> @_Z27__spirv_ConvertFToBF16INTELDv4_f(<4 x float> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <8 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <16 x i16> @_Z27__spirv_ConvertFToBF16INTELDv16_f(<16 x float> zeroinitializer)
; CHECK-REGULARIZED: call spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16 zeroext 0)
; CHECK-REGULARIZED: call spir_func <2 x float> @_Z27__spirv_ConvertBF16ToFINTELDv2_s(<2 x i16> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <3 x float> @_Z27__spirv_ConvertBF16ToFINTELDv3_s(<3 x i16> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <4 x float> @_Z27__spirv_ConvertBF16ToFINTELDv4_s(<4 x i16> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <8 x float> @_Z27__spirv_ConvertBF16ToFINTELDv8_s(<8 x i16> zeroinitializer)
; CHECK-REGULARIZED: call spir_func <16 x float> @_Z27__spirv_ConvertBF16ToFINTELDv16_s(<16 x i16> zeroinitializer)

; CHECK-SPIRV-DAG: TypeInt [[#Int16Ty:]] 16 0
; CHECK-SPIRV-DAG: TypeFloat [[#FloatTy:]] 32
; CHECK-SPIRV-DAG: TypeVector [[#VecFloat2:]] [[#FloatTy]] 2
; CHECK-SPIRV-DAG: TypeVector [[#VecInt162:]] [[#Int16Ty]] 2
; CHECK-SPIRV-DAG: TypeVector [[#VecFloat3:]] [[#FloatTy]] 3
; CHECK-SPIRV-DAG: TypeVector [[#VecInt163:]] [[#Int16Ty]] 3
; CHECK-SPIRV-DAG: TypeVector [[#VecFloat4:]] [[#FloatTy]] 4
; CHECK-SPIRV-DAG: TypeVector [[#VecInt164:]] [[#Int16Ty]] 4
; CHECK-SPIRV-DAG: TypeVector [[#VecFloat8:]] [[#FloatTy]] 8
; CHECK-SPIRV-DAG: TypeVector [[#VecInt168:]] [[#Int16Ty]] 8
; CHECK-SPIRV-DAG: TypeVector [[#VecFloat16:]] [[#FloatTy]] 16
; CHECK-SPIRV-DAG: TypeVector [[#VecInt1616:]] [[#Int16Ty]] 16

; CHECK-SPIRV: ConvertFToBF16INTEL [[#Int16Ty]]
; CHECK-SPIRV: ConvertFToBF16INTEL [[#VecInt162]]
; CHECK-SPIRV: ConvertFToBF16INTEL [[#VecInt163]]
; CHECK-SPIRV: ConvertFToBF16INTEL [[#VecInt164]]
; CHECK-SPIRV: ConvertFToBF16INTEL [[#VecInt168]]
; CHECK-SPIRV: ConvertFToBF16INTEL [[#VecInt1616]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#FloatTy]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#VecFloat2]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#VecFloat3]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#VecFloat4]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#VecFloat8]]
; CHECK-SPIRV: ConvertBF16ToFINTEL [[#VecFloat16]]

; CHECK-LLVM-SPV: call spir_func i16 @_Z27__spirv_ConvertFToBF16INTELf(float 0.000000e+00)
; CHECK-LLVM-SPV: call spir_func <2 x i16> @_Z27__spirv_ConvertFToBF16INTELDv2_f(<2 x float> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <3 x i16> @_Z27__spirv_ConvertFToBF16INTELDv3_f(<3 x float> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <4 x i16> @_Z27__spirv_ConvertFToBF16INTELDv4_f(<4 x float> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <8 x i16> @_Z27__spirv_ConvertFToBF16INTELDv8_f(<8 x float> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <16 x i16> @_Z27__spirv_ConvertFToBF16INTELDv16_f(<16 x float> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func float @_Z27__spirv_ConvertBF16ToFINTELs(i16 0)
; CHECK-LLVM-SPV: call spir_func <2 x float> @_Z27__spirv_ConvertBF16ToFINTELDv2_s(<2 x i16> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <3 x float> @_Z27__spirv_ConvertBF16ToFINTELDv3_s(<3 x i16> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <4 x float> @_Z27__spirv_ConvertBF16ToFINTELDv4_s(<4 x i16> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <8 x float> @_Z27__spirv_ConvertBF16ToFINTELDv8_s(<8 x i16> zeroinitializer)
; CHECK-LLVM-SPV: call spir_func <16 x float> @_Z27__spirv_ConvertBF16ToFINTELDv16_s(<16 x i16> zeroinitializer)

; CHECK-LLVM-CL20: call spir_func i16 @_Z32intel_convert_bfloat16_as_ushortf(float 0.000000e+00)
; CHECK-LLVM-CL20: call spir_func <2 x i16> @_Z34intel_convert_bfloat162_as_ushort2Dv2_f(<2 x float> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <3 x i16> @_Z34intel_convert_bfloat163_as_ushort3Dv3_f(<3 x float> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <4 x i16> @_Z34intel_convert_bfloat164_as_ushort4Dv4_f(<4 x float> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <8 x i16> @_Z34intel_convert_bfloat168_as_ushort8Dv8_f(<8 x float> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <16 x i16> @_Z36intel_convert_bfloat1616_as_ushort16Dv16_f(<16 x float> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func float @_Z31intel_convert_as_bfloat16_floats(i16 0)
; CHECK-LLVM-CL20: call spir_func <2 x float> @_Z33intel_convert_as_bfloat162_float2Dv2_s(<2 x i16> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <3 x float> @_Z33intel_convert_as_bfloat163_float3Dv3_s(<3 x i16> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <4 x float> @_Z33intel_convert_as_bfloat164_float4Dv4_s(<4 x i16> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <8 x float> @_Z33intel_convert_as_bfloat168_float8Dv8_s(<8 x i16> zeroinitializer)
; CHECK-LLVM-CL20: call spir_func <16 x float> @_Z35intel_convert_as_bfloat1616_float16Dv16_s(<16 x i16> zeroinitializer)

; ModuleID = 'kernel.cl'
source_filename = "kernel.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @f() {
entry:
  %call = call spir_func zeroext i16 @_Z32intel_convert_bfloat16_as_ushortf(float 0.000000e+00)
  %call1 = call spir_func <2 x i16> @_Z34intel_convert_bfloat162_as_ushort2Dv2_f(<2 x float> zeroinitializer)
  %call2 = call spir_func <3 x i16> @_Z34intel_convert_bfloat163_as_ushort3Dv3_f(<3 x float> zeroinitializer)
  %call3 = call spir_func <4 x i16> @_Z34intel_convert_bfloat164_as_ushort4Dv4_f(<4 x float> zeroinitializer)
  %call4 = call spir_func <8 x i16> @_Z34intel_convert_bfloat168_as_ushort8Dv8_f(<8 x float> zeroinitializer)
  %call5 = call spir_func <16 x i16> @_Z36intel_convert_bfloat1616_as_ushort16Dv16_f(<16 x float> zeroinitializer)
  %call6 = call spir_func float @_Z31intel_convert_as_bfloat16_floatt(i16 zeroext 0)
  %call7 = call spir_func <2 x float> @_Z33intel_convert_as_bfloat162_float2Dv2_t(<2 x i16> zeroinitializer)
  %call8 = call spir_func <3 x float> @_Z33intel_convert_as_bfloat163_float3Dv3_t(<3 x i16> zeroinitializer)
  %call9 = call spir_func <4 x float> @_Z33intel_convert_as_bfloat164_float4Dv4_t(<4 x i16> zeroinitializer)
  %call10 = call spir_func <8 x float> @_Z33intel_convert_as_bfloat168_float8Dv8_t(<8 x i16> zeroinitializer)
  %call11 = call spir_func <16 x float> @_Z35intel_convert_as_bfloat1616_float16Dv16_t(<16 x i16> zeroinitializer)
  ret void
}

; Function Attrs: convergent
declare spir_func zeroext i16 @_Z32intel_convert_bfloat16_as_ushortf(float)

; Function Attrs: convergent
declare spir_func <2 x i16> @_Z34intel_convert_bfloat162_as_ushort2Dv2_f(<2 x float>)

; Function Attrs: convergent
declare spir_func <3 x i16> @_Z34intel_convert_bfloat163_as_ushort3Dv3_f(<3 x float>)

; Function Attrs: convergent
declare spir_func <4 x i16> @_Z34intel_convert_bfloat164_as_ushort4Dv4_f(<4 x float>)

; Function Attrs: convergent
declare spir_func <8 x i16> @_Z34intel_convert_bfloat168_as_ushort8Dv8_f(<8 x float>)

; Function Attrs: convergent
declare spir_func <16 x i16> @_Z36intel_convert_bfloat1616_as_ushort16Dv16_f(<16 x float>)

; Function Attrs: convergent
declare spir_func float @_Z31intel_convert_as_bfloat16_floatt(i16 zeroext)

; Function Attrs: convergent
declare spir_func <2 x float> @_Z33intel_convert_as_bfloat162_float2Dv2_t(<2 x i16>)

; Function Attrs: convergent
declare spir_func <3 x float> @_Z33intel_convert_as_bfloat163_float3Dv3_t(<3 x i16>)

; Function Attrs: convergent
declare spir_func <4 x float> @_Z33intel_convert_as_bfloat164_float4Dv4_t(<4 x i16>)

; Function Attrs: convergent
declare spir_func <8 x float> @_Z33intel_convert_as_bfloat168_float8Dv8_t(<8 x i16>)

; Function Attrs: convergent
declare spir_func <16 x float> @_Z35intel_convert_as_bfloat1616_float16Dv16_t(<16 x i16>)

!opencl.ocl.version = !{!0}

!0 = !{i32 2, i32 0}
