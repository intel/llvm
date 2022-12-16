; RUN: sycl-post-link -split-esimd -lower-esimd -O0 -S %s -o %t.table
; RUN: FileCheck %s -input-file=%t_esimd_0.ll
; This test checks that IR code below can be successfully processed by
; sycl-post-link. In this IR no extractelement instruction and no casting are used

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInGlobalOffset = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInNumWorkgroups = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupSize = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInWorkgroupId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32
@__spirv_BuiltInSubgroupLocalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant i32, align 4
@__spirv_BuiltInSubgroupSize = external dso_local local_unnamed_addr addrspace(1) constant i32, align 4
@__spirv_BuiltInSubgroupMaxSize = external dso_local local_unnamed_addr addrspace(1) constant i32, align 4

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @ESIMD_kernel() #0 !sycl_explicit_simd !3 {
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32
  %conv = trunc i64 %0 to i32
  ret void
}
; CHECK: define dso_local spir_kernel void @ESIMD_kernel()
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.0 to i64
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.0 to i64
; CHECK: %Res.llvm.genx.group.id.x = call i32 @llvm.genx.group.id.x()
; CHECK: %Res.llvm.genx.group.id.x.cast.ty = zext i32 %Res.llvm.genx.group.id.x to i64
; CHECK: %mul = mul i64 %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty, %Res.llvm.genx.group.id.x.cast.ty
; CHECK: %add = add i64 %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty, %mul
; CHECK: %conv = trunc i64 %add to i32
; CHECK: }

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalInvocationId_x(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.0 to i64
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.0 to i64
; CHECK: %Res.llvm.genx.group.id.x = call i32 @llvm.genx.group.id.x()
; CHECK: %Res.llvm.genx.group.id.x.cast.ty = zext i32 %Res.llvm.genx.group.id.x to i64
; CHECK: %mul = mul i64 %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty, %Res.llvm.genx.group.id.x.cast.ty
; CHECK: %add = add i64 %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty, %mul
; CHECK: store i64 %add, i64 addrspace(1)* %0, align 8


; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalInvocationId_y(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 1), align 8
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.1 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 1
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.1.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.1 to i64
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.1 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 1
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.1.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.1 to i64
; CHECK: %Res.llvm.genx.group.id.y = call i32 @llvm.genx.group.id.y()
; CHECK: %Res.llvm.genx.group.id.y.cast.ty = zext i32 %Res.llvm.genx.group.id.y to i64
; CHECK: %mul = mul i64 %Res.llvm.genx.local.size.v3i32.ext.1.cast.ty, %Res.llvm.genx.group.id.y.cast.ty
; CHECK: %add = add i64 %Res.llvm.genx.local.id.v3i32.ext.1.cast.ty, %mul
; CHECK: store i64 %add, i64 addrspace(1)* %0, align 8


; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalInvocationId_z(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize)#0 !sycl_explicit_simd !3 {
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalInvocationId, i64 0, i64 2), align 16
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.2 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 2
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.2.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.2 to i64
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.2 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 2
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.2.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.2 to i64
; CHECK: %Res.llvm.genx.group.id.z = call i32 @llvm.genx.group.id.z()
; CHECK: %Res.llvm.genx.group.id.z.cast.ty = zext i32 %Res.llvm.genx.group.id.z to i64
; CHECK: %mul = mul i64 %Res.llvm.genx.local.size.v3i32.ext.2.cast.ty, %Res.llvm.genx.group.id.z.cast.ty
; CHECK: %add = add i64 %Res.llvm.genx.local.id.v3i32.ext.2.cast.ty, %mul
; CHECK: store i64 %add, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalSize_x(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize)#0 !sycl_explicit_simd !3 {
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalSize, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.0 to i64
; CHECK: %Res.llvm.genx.group.count.v3i32 = call <3 x i32> @llvm.genx.group.count.v3i32()
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.group.count.v3i32, i32 0
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.group.count.v3i32.ext.0 to i64
; CHECK: %mul = mul i64 %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty, %Res.llvm.genx.group.count.v3i32.ext.0.cast.ty
; CHECK: store i64 %mul, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalSize_y(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize)#0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalSize, i64 0, i64 1), align 8
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.1 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 1
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.1.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.1 to i64
; CHECK: %Res.llvm.genx.group.count.v3i32 = call <3 x i32> @llvm.genx.group.count.v3i32()
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.1 = extractelement <3 x i32> %Res.llvm.genx.group.count.v3i32, i32 1
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.1.cast.ty = zext i32 %Res.llvm.genx.group.count.v3i32.ext.1 to i64
; CHECK: %mul = mul i64 %Res.llvm.genx.local.size.v3i32.ext.1.cast.ty, %Res.llvm.genx.group.count.v3i32.ext.1.cast.ty
; CHECK: store i64 %mul, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalSize_z(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize)#0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalSize, i64 0, i64 2), align 16
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.2 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 2
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.2.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.2 to i64
; CHECK: %Res.llvm.genx.group.count.v3i32 = call <3 x i32> @llvm.genx.group.count.v3i32()
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.2 = extractelement <3 x i32> %Res.llvm.genx.group.count.v3i32, i32 2
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.2.cast.ty = zext i32 %Res.llvm.genx.group.count.v3i32.ext.2 to i64
; CHECK: %mul = mul i64 %Res.llvm.genx.local.size.v3i32.ext.2.cast.ty, %Res.llvm.genx.group.count.v3i32.ext.2.cast.ty
; CHECK: store i64 %mul, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalOffset_x(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalOffset, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: store i64 0, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalOffset_y(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalOffset, i64 0, i64 1), align 8
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: store i64 0, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_GlobalOffset_z(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInGlobalOffset, i64 0, i64 2), align 16
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: store i64 0, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_NumWorkgroups_x(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInNumWorkgroups, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.group.count.v3i32 = call <3 x i32> @llvm.genx.group.count.v3i32()
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.group.count.v3i32, i32 0
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.group.count.v3i32.ext.0 to i64
; CHECK: store i64 %Res.llvm.genx.group.count.v3i32.ext.0.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_NumWorkgroups_y(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInNumWorkgroups, i64 0, i64 1), align 8
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.group.count.v3i32 = call <3 x i32> @llvm.genx.group.count.v3i32()
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.1 = extractelement <3 x i32> %Res.llvm.genx.group.count.v3i32, i32 1
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.1.cast.ty = zext i32 %Res.llvm.genx.group.count.v3i32.ext.1 to i64
; CHECK: store i64 %Res.llvm.genx.group.count.v3i32.ext.1.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_NumWorkgroups_z(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInNumWorkgroups, i64 0, i64 2), align 16
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.group.count.v3i32 = call <3 x i32> @llvm.genx.group.count.v3i32()
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.2 = extractelement <3 x i32> %Res.llvm.genx.group.count.v3i32, i32 2
; CHECK: %Res.llvm.genx.group.count.v3i32.ext.2.cast.ty = zext i32 %Res.llvm.genx.group.count.v3i32.ext.2 to i64
; CHECK: store i64 %Res.llvm.genx.group.count.v3i32.ext.2.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_WorkgroupSize_x(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.0 to i64
; CHECK: store i64 %Res.llvm.genx.local.size.v3i32.ext.0.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_WorkgroupSize_y(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize, i64 0, i64 1), align 8
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.1 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 1
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.1.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.1 to i64
; CHECK: store i64 %Res.llvm.genx.local.size.v3i32.ext.1.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_WorkgroupSize_z(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupSize, i64 0, i64 2), align 16
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.size.v3i32 = call <3 x i32> @llvm.genx.local.size.v3i32()
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.2 = extractelement <3 x i32> %Res.llvm.genx.local.size.v3i32, i32 2
; CHECK: %Res.llvm.genx.local.size.v3i32.ext.2.cast.ty = zext i32 %Res.llvm.genx.local.size.v3i32.ext.2 to i64
; CHECK: store i64 %Res.llvm.genx.local.size.v3i32.ext.2.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_WorkgroupId_x(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %0 = call i64 addrspace(1)* @llvm.genx.address.convert.p1i64.p1i8(i8 addrspace(1)* %_arg_DoNotOptimize)
; CHECK: %Res.llvm.genx.group.id.x = call i32 @llvm.genx.group.id.x()
; CHECK: %Res.llvm.genx.group.id.x.cast.ty = zext i32 %Res.llvm.genx.group.id.x to i64
; CHECK: store i64 %Res.llvm.genx.group.id.x.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_WorkgroupId_y(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId, i64 0, i64 1), align 8
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.group.id.y = call i32 @llvm.genx.group.id.y()
; CHECK: %Res.llvm.genx.group.id.y.cast.ty = zext i32 %Res.llvm.genx.group.id.y to i64
; CHECK: store i64 %Res.llvm.genx.group.id.y.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_WorkgroupId_z(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId, i64 0, i64 2), align 16
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.group.id.z = call i32 @llvm.genx.group.id.z()
; CHECK: %Res.llvm.genx.group.id.z.cast.ty = zext i32 %Res.llvm.genx.group.id.z to i64
; CHECK: store i64 %Res.llvm.genx.group.id.z.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_LocalInvocationId_x(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.0 to i64
; CHECK: store i64 %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_LocalInvocationId_y(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId, i64 0, i64 1), align 8
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.1 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 1
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.1.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.1 to i64
; CHECK: store i64 %Res.llvm.genx.local.id.v3i32.ext.1.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_LocalInvocationId_z(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId, i64 0, i64 2), align 16
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.2 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 2
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.2.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.2 to i64
; CHECK: store i64 %Res.llvm.genx.local.id.v3i32.ext.2.cast.ty, i64 addrspace(1)* %0, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_LocalInvocationId_xyz(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimizeXYZ) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimizeXYZ, align 8
  %1 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId, i64 0, i64 1), align 8
  %arrayidx4.i2 = getelementptr inbounds i64, i64 addrspace(1)* %_arg_DoNotOptimizeXYZ, i64 1
  store i64 %1, i64 addrspace(1)* %arrayidx4.i2, align 8
  %2 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInLocalInvocationId, i64 0, i64 2), align 16
  %arrayidx7.i3 = getelementptr inbounds i64, i64 addrspace(1)* %_arg_DoNotOptimizeXYZ, i64 2
  store i64 %2, i64 addrspace(1)* %arrayidx7.i3, align 8
  ret void
}
; CHECK: %Res.llvm.genx.local.id.v3i32 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i32, i32 0
; CHECK: %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i32.ext.0 to i64
; CHECK: store i64 %Res.llvm.genx.local.id.v3i32.ext.0.cast.ty, i64 addrspace(1)* %0, align 8
; CHECK: %Res.llvm.genx.local.id.v3i321 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i321.ext.1 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i321, i32 1
; CHECK: %Res.llvm.genx.local.id.v3i321.ext.1.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i321.ext.1 to i64
; CHECK: %arrayidx4.i2 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
; CHECK: store i64 %Res.llvm.genx.local.id.v3i321.ext.1.cast.ty, i64 addrspace(1)* %arrayidx4.i2, align 8
; CHECK: %Res.llvm.genx.local.id.v3i322 = call <3 x i32> @llvm.genx.local.id.v3i32()
; CHECK: %Res.llvm.genx.local.id.v3i322.ext.2 = extractelement <3 x i32> %Res.llvm.genx.local.id.v3i322, i32 2
; CHECK: %Res.llvm.genx.local.id.v3i322.ext.2.cast.ty = zext i32 %Res.llvm.genx.local.id.v3i322.ext.2 to i64
; CHECK: %arrayidx7.i3 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 2
; CHECK: store i64 %Res.llvm.genx.local.id.v3i322.ext.2.cast.ty, i64 addrspace(1)* %arrayidx7.i3, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_WorkgroupId_xyz(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimizeXYZ) #0 !sycl_explicit_simd !3{
entry:
  %0 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId, i64 0, i64 0), align 32
  store i64 %0, i64 addrspace(1)* %_arg_DoNotOptimizeXYZ, align 8
  %1 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId, i64 0, i64 1), align 8
  %arrayidx4.i2 = getelementptr inbounds i64, i64 addrspace(1)* %_arg_DoNotOptimizeXYZ, i64 1
  store i64 %1, i64 addrspace(1)* %arrayidx4.i2, align 8
  %2 = load i64, i64 addrspace(1)* getelementptr (<3 x i64>, <3 x i64> addrspace(1)* @__spirv_BuiltInWorkgroupId, i64 0, i64 2), align 16
  %arrayidx7.i3 = getelementptr inbounds i64, i64 addrspace(1)* %_arg_DoNotOptimizeXYZ, i64 2
  store i64 %2, i64 addrspace(1)* %arrayidx7.i3, align 8
  ret void
}
; CHECK: %Res.llvm.genx.group.id.x = call i32 @llvm.genx.group.id.x()
; CHECK: %Res.llvm.genx.group.id.x.cast.ty = zext i32 %Res.llvm.genx.group.id.x to i64
; CHECK: store i64 %Res.llvm.genx.group.id.x.cast.ty, i64 addrspace(1)* %0, align 8
; CHECK: %Res.llvm.genx.group.id.y = call i32 @llvm.genx.group.id.y()
; CHECK: %Res.llvm.genx.group.id.y.cast.ty = zext i32 %Res.llvm.genx.group.id.y to i64
; CHECK: %arrayidx4.i2 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 1
; CHECK: store i64 %Res.llvm.genx.group.id.y.cast.ty, i64 addrspace(1)* %arrayidx4.i2, align 8
; CHECK: %Res.llvm.genx.group.id.z = call i32 @llvm.genx.group.id.z()
; CHECK: %Res.llvm.genx.group.id.z.cast.ty = zext i32 %Res.llvm.genx.group.id.z to i64
; CHECK: %arrayidx7.i3 = getelementptr inbounds i64, i64 addrspace(1)* %0, i64 2
; CHECK: store i64 %Res.llvm.genx.group.id.z.cast.ty, i64 addrspace(1)* %arrayidx7.i3, align 8

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_SubgroupLocalInvocationId(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize, i32 addrspace(1)* noundef align 4 %_arg_DoNotOptimize32) #0 !sycl_explicit_simd !3 {
entry:
  %0 = load i32, i32 addrspace(1)* @__spirv_BuiltInSubgroupLocalInvocationId, align 4
  %conv.i = zext i32 %0 to i64
  store i64 %conv.i, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  %add.i = add i32 %0, 3
  store i32 %add.i, i32 addrspace(1)* %_arg_DoNotOptimize32, align 4
  ret void
}
; CHECK: %conv.i = zext i32 0 to i64
; CHECK: store i64 %conv.i, i64 addrspace(1)* %0, align 8
; CHECK: %add.i = add i32 0, 3
; CHECK: store i32 %add.i, i32 addrspace(1)* %1, align 4

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_SubgroupSize(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize, i32 addrspace(1)* noundef align 4 %_arg_DoNotOptimize32)#0 !sycl_explicit_simd !3{
entry:
  %0 = load i32, i32 addrspace(1)* @__spirv_BuiltInSubgroupSize, align 4
  %conv.i = zext i32 %0 to i64
  store i64 %conv.i, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  %add.i = add i32 %0, 7
  store i32 %add.i, i32 addrspace(1)* %_arg_DoNotOptimize32, align 4
  ret void
}
; CHECK: %conv.i = zext i32 1 to i64
; CHECK: store i64 %conv.i, i64 addrspace(1)* %0, align 8
; CHECK: %add.i = add i32 1, 7
; CHECK: store i32 %add.i, i32 addrspace(1)* %1, align 4

; Function Attrs: convergent norecurse
define dso_local spir_kernel void @kernel_SubgroupMaxSize(i64 addrspace(1)* noundef align 8 %_arg_DoNotOptimize, i32 addrspace(1)* noundef align 4 %_arg_DoNotOptimize32) #0 !sycl_explicit_simd !3 {
entry:
  %0 = load i32, i32 addrspace(1)* @__spirv_BuiltInSubgroupMaxSize, align 4
  %conv.i = zext i32 %0 to i64
  store i64 %conv.i, i64 addrspace(1)* %_arg_DoNotOptimize, align 8
  %add.i = add i32 %0, 9
  store i32 %add.i, i32 addrspace(1)* %_arg_DoNotOptimize32, align 4
  ret void
}
; CHECK: %conv.i = zext i32 1 to i64
; CHECK: store i64 %conv.i, i64 addrspace(1)* %0, align 8
; CHECK: %add.i = add i32 1, 9
; CHECK: store i32 %add.i, i32 addrspace(1)* %1, align 4

attributes #0 = { "sycl-module-id"="a.cpp" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 0, i32 100000}
!3 = !{}


