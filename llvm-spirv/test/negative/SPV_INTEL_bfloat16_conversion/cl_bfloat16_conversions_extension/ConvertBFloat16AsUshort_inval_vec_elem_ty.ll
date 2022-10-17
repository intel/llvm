; RUN: llvm-as %s -o %t.bc
; RUN: not --crash llvm-spirv %t.bc -o %t.spv 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: OpConvertBFloat16NAsUShortN must be of <N x i16> and take <N x float>

; ModuleID = 'kernel.cl'
source_filename = "kernel.cl"
target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir"

; Function Attrs: convergent noinline norecurse nounwind optnone
define dso_local spir_kernel void @f() {
entry:
  %call = call spir_func <2 x i32> @_Z34intel_convert_bfloat162_as_ushort2Dv2_f(<2 x double> zeroinitializer)
  ret void
}

; Function Attrs: convergent
declare spir_func <2 x i32> @_Z34intel_convert_bfloat162_as_ushort2Dv2_f(<2 x double>)

!opencl.ocl.version = !{!0}

!0 = !{i32 2, i32 0}
