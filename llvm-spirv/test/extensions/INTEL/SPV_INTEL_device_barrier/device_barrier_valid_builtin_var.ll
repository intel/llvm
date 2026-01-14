; RUN: llvm-as %s -o %t.bc

; Test with typed pointers:
; RUN: llvm-spirv %t.bc -o %t.spv --spirv-ext=+SPV_INTEL_device_barrier --spirv-max-version=1.0
;; TODO: Enable validation when SPV_INTEL_device_barrier supported.
; RUNx: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -o %t.spt --to-text
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.spv --spirv-target-env=SPV-IR -o %t.out.bc
; RUN: llvm-dis %t.out.bc -o - | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -spirv-text %t.out.bc --spirv-ext=+SPV_INTEL_device_barrier --spirv-max-version=1.0 -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; Test with untyped pointers
; RUN: llvm-spirv %t.bc -o %t.up.spv --spirv-ext=+SPV_KHR_untyped_pointers,+SPV_INTEL_device_barrier --spirv-max-version=1.0
;; TODO: Enable validation when SPV_INTEL_device_barrier supported.
; RUNx: spirv-val %t.up.spv
; RUN: llvm-spirv %t.up.spv -o %t.up.spt --to-text 
; RUN: FileCheck < %t.up.spt %s --check-prefix=CHECK-SPIRV

; RUN: llvm-spirv -r %t.up.spv --spirv-target-env=SPV-IR -o %t.up.out.bc
; RUN: llvm-dis %t.up.out.bc -o - | FileCheck %s --check-prefix=CHECK-SPV-IR
; RUN: llvm-spirv -spirv-text %t.up.out.bc --spirv-ext=+SPV_KHR_untyped_pointers,+SPV_INTEL_device_barrier --spirv-max-version=1.0 -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; Note: 6186 is DeviceBarrierValidINTEL
; CHECK-SPIRV: Decorate [[Id:[0-9]+]] BuiltIn 6186
; CHECK-SPIRV: {{(Variable|UntypedVariableKHR)}} {{[0-9]+}} [[Id:[0-9]+]]

target datalayout = "e-p:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir-unknown-unknown"

@__spirv_BuiltInDeviceBarrierValidINTEL = external addrspace(1) global i1

; Function Attrs: nounwind readnone
define spir_kernel void @f() #0 !kernel_arg_addr_space !0 !kernel_arg_access_qual !0 !kernel_arg_type !0 !kernel_arg_base_type !0 !kernel_arg_type_qual !0 {
entry:
  %0 = load i1, ptr addrspace(1) @__spirv_BuiltInDeviceBarrierValidINTEL, align 4
  ; CHECK-SPV-IR: %0 = call spir_func i1 @_Z38__spirv_BuiltInDeviceBarrierValidINTELv() #1
  ret void
}

attributes #0 = { alwaysinline nounwind readonly "less-precise-fpmad"="false" "no-frame-pointer-elim"="false" "no-infs-fp-math"="false" "no-nans-fp-math"="false" "no-realign-stack" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }

!0 = !{}
