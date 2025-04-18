; REQUIRES: pass-plugin
; UNSUPPORTED: target={{.*windows.*}}

; RUN: opt %load_spirv_lib -passes=spirv-to-ocl20 %s -S -o - | FileCheck %s --check-prefixes=CHECK-LLVM-OCL

%id = type { %arr }
%arr = type { [1 x i64] }

@__spirv_BuiltInGlobalInvocationId = external local_unnamed_addr addrspace(1) constant <3 x i64>

; Check that SPIR-V Builtin GenericCastToPtr and GenericCastToPtrExplicit can be transformed to OCL.

; CHECK-LLVM-OCL-LABEL: void @test(
; CHECK-LLVM-OCL: %[[VAR_1:.*]] = addrspacecast ptr addrspace(1) %idx to ptr addrspace(4)
; CHECK-LLVM-OCL-NEXT: %[[VAR_2:.*]] = addrspacecast ptr addrspace(3) %_arg_LocalA to ptr addrspace(4)
; CHECK-LLVM-OCL-NEXT: %[[VAR_3:.*]] = addrspacecast ptr %var to ptr addrspace(4)
; CHECK-LLVM-OCL-NEXT: addrspacecast ptr addrspace(4) %[[VAR_1]] to ptr addrspace(1)
; CHECK-LLVM-OCL-NEXT: addrspacecast ptr addrspace(4) %[[VAR_2]] to ptr addrspace(3)
; CHECK-LLVM-OCL-NEXT: addrspacecast ptr addrspace(4) %[[VAR_3]] to ptr
; CHECK-LLVM-OCL-NEXT: call spir_func ptr addrspace(1) @__to_global(ptr addrspace(4) %[[VAR_1]])
; CHECK-LLVM-OCL-NEXT: call spir_func ptr addrspace(3) @__to_local(ptr addrspace(4) %[[VAR_2]])
; CHECK-LLVM-OCL-NEXT: call spir_func ptr @__to_private(ptr addrspace(4) %[[VAR_3]])

define spir_kernel void @test(ptr addrspace(1) %_arg_GlobalA, ptr byval(%id) %_arg_GlobalId, ptr addrspace(3) %_arg_LocalA) {
entry:
  %var = alloca i32
  %p0 = load i64, ptr %_arg_GlobalId
  %add = getelementptr inbounds i32, ptr addrspace(1) %_arg_GlobalA, i64 %p0
  %p2 = load i64, ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId
  %idx = getelementptr inbounds i32, ptr addrspace(1) %add, i64 %p2
  %var1 = addrspacecast ptr addrspace(1) %idx to ptr addrspace(4)
  %var2 = addrspacecast ptr addrspace(3) %_arg_LocalA to ptr addrspace(4)
  %var3 = addrspacecast ptr %var to ptr addrspace(4)
  %G = call spir_func ptr addrspace(1) @_Z33__spirv_GenericCastToPtr_ToGlobalPvi(ptr addrspace(4) %var1, i32 5)
  %L = call spir_func ptr addrspace(3) @_Z32__spirv_GenericCastToPtr_ToLocalPvi(ptr addrspace(4) %var2, i32 4)
  %P = call spir_func ptr @_Z34__spirv_GenericCastToPtr_ToPrivatePvi(ptr addrspace(4) %var3, i32 7)
  %GE = call spir_func ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4) %var1, i32 5)
  %LE = call spir_func ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4) %var2, i32 4)
  %PE = call spir_func ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4) %var3, i32 7)
  ret void
}

declare spir_func ptr addrspace(1) @_Z33__spirv_GenericCastToPtr_ToGlobalPvi(ptr addrspace(4), i32)
declare spir_func ptr addrspace(3) @_Z32__spirv_GenericCastToPtr_ToLocalPvi(ptr addrspace(4), i32)
declare spir_func ptr @_Z34__spirv_GenericCastToPtr_ToPrivatePvi(ptr addrspace(4), i32)
declare spir_func ptr addrspace(1) @_Z41__spirv_GenericCastToPtrExplicit_ToGlobalPvi(ptr addrspace(4), i32)
declare spir_func ptr addrspace(3) @_Z40__spirv_GenericCastToPtrExplicit_ToLocalPvi(ptr addrspace(4), i32)
declare spir_func ptr @_Z42__spirv_GenericCastToPtrExplicit_ToPrivatePvi(ptr addrspace(4), i32)
