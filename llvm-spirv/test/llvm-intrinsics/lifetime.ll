; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.spv.bc
; RUN: llvm-dis < %t.spv.bc | FileCheck %s --check-prefix=CHECK-LLVM

; Verify that we have valid SPV and the same output LLVM IR when using untyped pointers.
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_KHR_untyped_pointers -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv -r %t.spv -o %t.spv.bc
; RUN: llvm-dis < %t.spv.bc | FileCheck %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#SimpleF:]] "lifetime_simple"
; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#SizedF:]] "lifetime_sized"
; CHECK-SPIRV-DAG: EntryPoint [[#]] [[#GenericF:]] "lifetime_generic"
; CHECK-SPIRV-DAG: TypeStruct [[#StructTy:]] [[#]]
; CHECK-SPIRV-COUNT-1: TypePointer [[#PrivatePtrTy:]] [[#StructTy]]

; CHECK-SPIRV: Function [[#]] [[#SimpleF:]]
; CHECK-SPIRV: LifetimeStart [[#Tmp:]] 0
; CHECK-SPIRV: LifetimeStop [[#Tmp]] 0

; CHECK-SPIRV: Function [[#]] [[#SizedF:]]
; CHECK-SPIRV: LifetimeStart [[#Tmp:]] 1
; CHECK-SPIRV: Bitcast [[#]] [[#Cast:]]
; CHECK-SPIRV: LifetimeStop [[#Cast]] 1

; CHECK-SPIRV: Function [[#]] [[#GenericF:]]
; CHECK-SPIRV: Variable [[#PrivatePtrTy]] [[#Var:]] 7
; CHECK-SPIRV: PtrCastToGeneric [[#]] [[#Cast1:]] [[#Var]]
; CHECK-SPIRV: Bitcast [[#]] [[#Cast2:]] [[#Cast1]]
; CHECK-SPIRV: GenericCastToPtr [[#]] [[#Cast3:]] [[#Cast2]]
; CHECK-SPIRV: LifetimeStart [[#Cast3]] 1
; CHECK-SPIRV: GenericCastToPtr [[#]] [[#Cast4:]]
; CHECK-SPIRV: LifetimeStop [[#Cast4]] 1

; CHECK-LLVM-LABEL: lifetime_simple
; CHECK-LLVM: %[[tmp1:[0-9]+]] = bitcast ptr %{{[0-9]+}} to ptr
; CHECK-LLVM: call void @llvm.lifetime.start.p0(i64 -1, ptr %[[tmp1]])
; CHECK-LLVM: call void @llvm.lifetime.end.p0(i64 -1, ptr %[[tmp1]])

; CHECK-LLVM-LABEL: lifetime_sized
; CHECK-LLVM: call void @llvm.lifetime.start.p0(i64 1, ptr %[[#]])
; CHECK-LLVM: call void @llvm.lifetime.end.p0(i64 1, ptr %[[#]])

; CHECK-LLVM-LABEL: lifetime_generic
; CHECK-LLVM: %[[#Alloca:]] = alloca %class.anon
; CHECK-LLVM: %[[#Cast1:]] = addrspacecast ptr %[[#Alloca]] to ptr addrspace(4)
; CHECK-LLVM: %[[#Cast2:]] = bitcast ptr addrspace(4) %[[#Cast1]] to ptr addrspace(4)
; CHECK-LLVM: %[[#Cast3:]] = addrspacecast ptr addrspace(4) %[[#Cast2:]] to ptr
; CHECK-LLVM: call void @llvm.lifetime.start.p0(i64 1, ptr %[[#Cast3]])
; CHECK-LLVM: %[[#Cast4:]] = addrspacecast ptr addrspace(4) %[[#]] to ptr
; CHECK-LLVM: call void @llvm.lifetime.end.p0(i64 1, ptr %[[#Cast4]])

; ModuleID = 'main'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%class.anon = type { i8 }

; Function Attrs: nounwind
define spir_kernel void @lifetime_simple(i32 addrspace(1)* nocapture %res, i32 addrspace(1)* nocapture %lhs, i32 addrspace(1)* nocapture %rhs) #0 !kernel_arg_addr_space !1 !kernel_arg_access_qual !2 !kernel_arg_type !3 !kernel_arg_base_type !5 !kernel_arg_type_qual !4 {
  %1 = alloca i32
  %2 = call spir_func i64 @_Z13get_global_idj(i32 0) #1
  %3 = shl i64 %2, 32
  %4 = ashr exact i64 %3, 32
  %5 = getelementptr inbounds i32, i32 addrspace(1)* %lhs, i64 %4
  %6 = load i32, i32 addrspace(1)* %5, align 4
  %7 = getelementptr inbounds i32, i32 addrspace(1)* %rhs, i64 %4
  %8 = load i32, i32 addrspace(1)* %7, align 4
  %9 = sub i32 %6, %8
  %10 = bitcast i32* %1 to i8*
  call void @llvm.lifetime.start.p0i8(i64 -1, i8* %10)
  store i32 %9, i32* %1
  %11 = load i32, i32* %1
  call void @llvm.lifetime.end.p0i8(i64 -1, i8* %10)
  %12 = getelementptr inbounds i32, i32 addrspace(1)* %res, i64 %4
  store i32 %11, i32 addrspace(1)* %12, align 4
  ret void
}

define spir_kernel void @lifetime_sized() #0 !kernel_arg_addr_space !8 !kernel_arg_access_qual !8 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !8 {
entry:
  %0 = alloca %class.anon, align 1
  %1 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.start.p0i8(i64 1, i8* %1) #0
  call spir_func void @foo(%class.anon* %0)
  %2 = bitcast %class.anon* %0 to i8*
  call void @llvm.lifetime.end.p0i8(i64 1, i8* %2) #0
  ret void
}

declare spir_func void @foo(%class.anon* %this) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #0

; Function Attrs: nounwind readnone
declare spir_func i64 @_Z13get_global_idj(i32) #1

define spir_kernel void @lifetime_generic() #0 !kernel_arg_addr_space !8 !kernel_arg_access_qual !8 !kernel_arg_type !8 !kernel_arg_base_type !8 !kernel_arg_type_qual !8 {
entry:
  %0 = alloca %class.anon, align 1, addrspace(4)
  %1 = bitcast %class.anon addrspace(4)* %0 to i8 addrspace(4)*
  call void @llvm.lifetime.start.p4i8(i64 1, i8 addrspace(4)* %1) #0
  call spir_func void @boo(%class.anon addrspace(4)* %0)
  %2 = bitcast %class.anon addrspace(4)* %0 to i8 addrspace(4)*
  call void @llvm.lifetime.end.p4i8(i64 1, i8 addrspace(4)* %2) #0
  ret void
}

declare spir_func void @boo(%class.anon addrspace(4)* %this) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.start.p4i8(i64 immarg, i8 addrspace(4)* nocapture) #0

; Function Attrs: nounwind
declare void @llvm.lifetime.end.p4i8(i64 immarg, i8 addrspace(4)* nocapture) #0


attributes #0 = { nounwind }
attributes #1 = { nounwind readnone }

!opencl.enable.FP_CONTRACT = !{}
!spirv.Source = !{!6}
!opencl.spir.version = !{!7}
!opencl.ocl.version = !{!7}
!opencl.used.extensions = !{!8}
!opencl.used.optional.core.features = !{!8}
!spirv.Generator = !{!9}

!1 = !{i32 1, i32 1, i32 1}
!2 = !{!"none", !"none", !"none"}
!3 = !{!"int*", !"int*", !"int*"}
!4 = !{!"", !"", !""}
!5 = !{!"int*", !"int*", !"int*"}
!6 = !{i32 3, i32 102000}
!7 = !{i32 1, i32 2}
!8 = !{}
!9 = !{i16 7, i16 0}
