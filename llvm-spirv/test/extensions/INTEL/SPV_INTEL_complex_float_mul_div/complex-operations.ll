; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_complex_float_mul_div
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv %t.spv -o %t.rev.bc -r
; RUN: llvm-dis %t.rev.bc -o %t.rev.ll
; RUN: FileCheck < %t.rev.ll %s --check-prefix=CHECK-LLVM

; RUN: not llvm-spirv %t.bc 2>&1 | FileCheck %s --check-prefix=CHECK-ERROR

; CHECK-ERROR: RequiresExtension: Feature requires the following SPIR-V extension:
; CHECK-ERROR-NEXT: SPV_INTEL_complex_float_mul_div

; CHECK-SPIRV: Capability ComplexFloatMulDivINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_complex_float_mul_div"
; CHECK-SPIRV: TypeFloat [[#TyFloat32:]] 32  
; CHECK-SPIRV: TypeVector [[#TyVec2Float32:]] [[#TyFloat32]] 2
; CHECK-SPIRV: ComplexFDivINTEL [[#TyVec2Float32]] [[#]] [[#]] [[#]]
; CHECK-SPIRV: ComplexFMulINTEL [[#TyVec2Float32]] [[#]] [[#]] [[#]]

; CHECK-LLVM: call spir_func <2 x float> @_Z24__spirv_ComplexFDivINTEL{{.*}}(<2 x float>{{.*}}, <2 x float>{{.*}})
; CHECK-LLVM: call spir_func <2 x float> @_Z24__spirv_ComplexFMulINTEL{{.*}}(<2 x float>{{.*}}, <2 x float>{{.*}})

; ModuleID = 'complex-operations'
target datalayout = "e-p:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"
target triple = "spir-unknown-unknown"

%"struct.std::complex" = type { %structtype }
%structtype = type { float, float }

; Function Attrs: nounwind
define spir_func void @_Z19cmul_kernel_complexPSt7complexIfES1_S1_(ptr noalias nocapture readonly %a, ptr noalias nocapture readonly %b, ptr noalias nocapture %c) #0 {
entry:
  %0 = load <2 x float>, ptr %a, align 4
  %1 = load <2 x float>, ptr %b, align 4
  %2 = call spir_func <2 x float> @_Z24__spirv_ComplexFDivINTELDv2_fS_(<2 x float> %0, <2 x float> %1) #0
  store <2 x float> %2, ptr %c, align 4
  %3 = call spir_func <2 x float> @_Z24__spirv_ComplexFMulINTELDv2_fS_(<2 x float> %0, <2 x float> %1) #0
  store <2 x float> %3, ptr %c, align 4
  ret void
}

; Function Attrs: nounwind
declare spir_func <2 x float> @_Z24__spirv_ComplexFDivINTELDv2_fS_(<2 x float>, <2 x float>) #0

; Function Attrs: nounwind
declare spir_func <2 x float> @_Z24__spirv_ComplexFMulINTELDv2_fS_(<2 x float>, <2 x float>) #0

attributes #0 = { nounwind }

!spirv.MemoryModel = !{!0}
!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!1}
!opencl.spir.version = !{!0}
!opencl.ocl.version = !{!1}
!opencl.used.extensions = !{!2}
!opencl.used.optional.core.features = !{!2}
!spirv.Generator = !{!3}

!0 = !{i32 1, i32 2}
!1 = !{i32 0, i32 0}
!2 = !{}
!3 = !{i16 6, i16 14}
