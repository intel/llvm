; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.rev.bc
; RUN: llvm-dis < %t.rev.bc | FileCheck %s --check-prefix=CHECK-LLVM
; RUN: %if spirv-backend %{ llc -O0 -mtriple=spirv64-unknown-unknown -filetype=obj %s -o %t.llc.spv %}
; RUN: %if spirv-backend %{ llvm-spirv -r %t.llc.spv -o %t.llc.rev.bc %}
; RUN: %if spirv-backend %{ llvm-dis %t.llc.rev.bc -o %t.llc.rev.ll %}
; RUN: %if spirv-backend %{ FileCheck %s --check-prefix=CHECK-LLVM < %t.llc.rev.ll %}

; Check saturation conversion is translated when there is forward declaration
; of SPIRV entry.

; CHECK-SPIRV: Decorate [[SAT:[0-9]+]] SaturatedConversion
; CHECK-SPIRV: ConvertFToU {{[0-9]+}} [[SAT]]

; CHECK-LLVM: convert_uchar_satf

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spir64"

declare spir_func zeroext i8 @_Z30__spirv_ConvertFToU_Ruchar_satf(float)

define spir_func void @forward(float %val, ptr addrspace(1) %dst) {
entry:
  br label %for.cond

for.cond:                                         ; preds = %for.body, %entry
  %new_val.0 = phi i8 [ undef, %entry ], [ %call1, %for.body ]
  %i.0 = phi i32 [ 0, %entry ], [ %inc, %for.body ]
  %cmp = icmp ult i32 %i.0, 1
  br i1 %cmp, label %for.body, label %for.end

for.body:                                         ; preds = %for.cond
  %call1 = call spir_func zeroext i8 @_Z30__spirv_ConvertFToU_Ruchar_satf(float noundef %val)
  %inc = add i32 %i.0, 1
  br label %for.cond

for.end:                                          ; preds = %for.cond
  store i8 %new_val.0, ptr addrspace(1) %dst, align 1
  ret void
}
