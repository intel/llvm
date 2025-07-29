; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv -spirv-text %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM-OCL
; RUN: llvm-spirv -r --spirv-target-env=SPV-IR %t.spv -o - | llvm-dis -o - | FileCheck %s --check-prefix=CHECK-LLVM-SPV

; RUN: llvm-spirv -spirv-text --spirv-max-version=1.5 %t.bc -o - | FileCheck %s --check-prefix=CHECK-SPIRV-NEG

; CHECK-SPIRV: Decorate [[#FPDec1:]] FPFastMathMode 3
; CHECK-SPIRV: Decorate [[#FPDec2:]] FPFastMathMode 2
; CHECK-SPIRV: Decorate [[#FPDec3:]] FPFastMathMode 3
; CHECK-SPIRV: Decorate [[#FPDec4:]] FPFastMathMode 16
; CHECK-SPIRV: ExtInst [[#]] [[#FPDec1]] [[#]] fmax [[#]] [[#]]
; CHECK-SPIRV: ExtInst [[#]] [[#FPDec2]] [[#]] fmin [[#]] [[#]]
; CHECK-SPIRV: ExtInst [[#]] [[#FPDec3]] [[#]] ldexp [[#]] [[#]]
; CHECK-SPIRV: ExtInst [[#]] [[#FPDec4]] [[#]] fmax [[#]] [[#]]

; CHECK-SPIRV-NEG-NOT: Decorate [[#]] FPFastMathMode [[#]]

; CHECK-LLVM-OCL: call nnan ninf spir_func float @_Z4fmaxff(float %[[#]], float %[[#]])
; CHECK-LLVM-OCL: call ninf spir_func float @_Z4fminff(float %[[#]], float %[[#]])
; CHECK-LLVM-OCL: call nnan ninf spir_func float @_Z5ldexpfi(float %[[#]], i32 %[[#]])
; CHECK-LLVM-OCL: call fast spir_func float @_Z4fmaxff(float %[[#]], float %[[#]])

; CHECK-LLVM-SPV: call nnan ninf spir_func float @_Z16__spirv_ocl_fmaxff(float %[[#]], float %[[#]])
; CHECK-LLVM-SPV: call ninf spir_func float @_Z16__spirv_ocl_fminff(float %[[#]], float %[[#]])
; CHECK-LLVM-SPV: call nnan ninf spir_func float @_Z17__spirv_ocl_ldexpfi(float %[[#]], i32 %[[#]])
; CHECK-LLVM-SPV: call fast spir_func float @_Z16__spirv_ocl_fmaxff(float %[[#]], float %[[#]])

; ModuleID = 'test.bc'
source_filename = "test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

declare dso_local spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf), float noundef nofpclass(nan inf)) local_unnamed_addr

declare dso_local spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fminff(float noundef nofpclass(inf), float noundef nofpclass(nan inf)) local_unnamed_addr
  
declare dso_local spir_func noundef nofpclass(nan inf) float @_Z17__spirv_ocl_ldexpfi(float noundef nofpclass(nan inf), i32 noundef)

define weak_odr dso_local spir_kernel void @nofpclass_all(ptr addrspace(1) noundef align 4 %_arg_data, ptr addrspace(1) noundef align 4 %_arg_dat1, ptr addrspace(1) noundef align 4 %_arg_dat2) local_unnamed_addr  {
entry:
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %_arg_data, i64 %0
  %arrayidx3.i = getelementptr inbounds float, ptr addrspace(1) %_arg_dat1, i64 %0
  %cmp.i = icmp ult i64 %0, 2147483648
  %arrayidx5.i = getelementptr inbounds float, ptr addrspace(1) %_arg_dat2, i64 %0
  %1 = load float, ptr addrspace(1) %arrayidx3.i, align 4
  %2 = load float, ptr addrspace(1) %arrayidx5.i, align 4
  %call.i.i = tail call spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  store float %call.i.i, ptr addrspace(1) %arrayidx.i, align 4
  ret void
}

define weak_odr dso_local spir_kernel void @nofpclass_part(ptr addrspace(1) noundef align 4 %_arg_data, ptr addrspace(1) noundef align 4 %_arg_dat1, ptr addrspace(1) noundef align 4 %_arg_dat2) local_unnamed_addr  {
entry:
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %_arg_data, i64 %0
  %arrayidx3.i = getelementptr inbounds float, ptr addrspace(1) %_arg_dat1, i64 %0
  %cmp.i = icmp ult i64 %0, 2147483648
  %arrayidx5.i = getelementptr inbounds float, ptr addrspace(1) %_arg_dat2, i64 %0
  %1 = load float, ptr addrspace(1) %arrayidx3.i, align 4
  %2 = load float, ptr addrspace(1) %arrayidx5.i, align 4
  %call.i.i = tail call spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fminff(float noundef nofpclass(inf) %1, float noundef nofpclass(nan inf) %2)
  store float %call.i.i, ptr addrspace(1) %arrayidx.i, align 4
  ret void
}

define weak_odr dso_local spir_kernel void @nofpclass_int(ptr addrspace(1) noundef align 4 %_arg_data, ptr addrspace(1) noundef align 4 %_arg_dat1, ptr addrspace(1) noundef align 4 %_arg_dat2) local_unnamed_addr  {
entry:
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %_arg_data, i64 %0
  %arrayidx3.i = getelementptr inbounds float, ptr addrspace(1) %_arg_dat1, i64 %0
  %cmp.i = icmp ult i64 %0, 2147483648
  %arrayidx5.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_dat2, i64 %0
  %1 = load float, ptr addrspace(1) %arrayidx3.i, align 4
  %2 = load i32, ptr addrspace(1) %arrayidx5.i, align 4
  %call.i.i = tail call spir_func noundef nofpclass(nan inf) float @_Z17__spirv_ocl_ldexpfi(float noundef nofpclass(inf) %1, i32 noundef %2)
  store float %call.i.i, ptr addrspace(1) %arrayidx.i, align 4
  ret void
}

define weak_odr dso_local spir_kernel void @nofpclass_fast(ptr addrspace(1) noundef align 4 %_arg_data, ptr addrspace(1) noundef align 4 %_arg_dat1, ptr addrspace(1) noundef align 4 %_arg_dat2) local_unnamed_addr  {
entry:
  %0 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId, align 32
  %arrayidx.i = getelementptr inbounds float, ptr addrspace(1) %_arg_data, i64 %0
  %arrayidx3.i = getelementptr inbounds float, ptr addrspace(1) %_arg_dat1, i64 %0
  %cmp.i = icmp ult i64 %0, 2147483648
  %arrayidx5.i = getelementptr inbounds float, ptr addrspace(1) %_arg_dat2, i64 %0
  %1 = load float, ptr addrspace(1) %arrayidx3.i, align 4
  %2 = load float, ptr addrspace(1) %arrayidx5.i, align 4
  %call.i.i = tail call fast spir_func noundef nofpclass(nan inf) float @_Z16__spirv_ocl_fmaxff(float noundef nofpclass(nan inf) %1, float noundef nofpclass(nan inf) %2)
  store float %call.i.i, ptr addrspace(1) %arrayidx.i, align 4
  ret void
}
