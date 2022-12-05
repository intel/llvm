; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
;
; This tests that translator is able to convert the following recursive data
; type which contains function pointers:
;
; typedef void(*fp_t)(struct Desc *);
; typedef Desc *(*fp2_t)();
;
; struct Type;
;
; struct Desc {
;   int B;
;   Type *Ptr;
; };
;
; struct Type {
;   int A;
;   fp_t FP;
; };
;
; __kernel void foo() {
;   Type T;
;   fp2_t ptr;
; }
;
; Without proper recursive types detection, the translator crashes while trying
; to translate this LLVM IR example with the following error:
;
; virtual SPIRV::SPIRVEntry* SPIRV::SPIRVModuleImpl::getEntry(SPIRV::SPIRVId) const: Assertion `Loc != IdEntryMap.end() && "Id is not in map"' failed.
;
; CHECK-SPIRV: TypeForwardPointer [[TYPEPTRTY:[0-9]+]]
; CHECK-SPIRV: TypeInt [[INTTY:[0-9]+]] 32 0
; CHECK-SPIRV: TypeVoid [[VOIDTY:[0-9]+]]
; CHECK-SPIRV: TypeStruct [[DESCTY:[0-9]+]] [[INTTY]] [[TYPEPTRTY]]
; CHECK-SPIRV: TypePointer [[DESCPTRTY:[0-9]+]] {{[0-9]+}} [[DESCTY]]
; CHECK-SPIRV: TypeFunction [[FTY:[0-9]+]] [[VOIDTY]] [[DESCPTRTY]]
; CHECK-SPIRV: TypePointer [[FPTRTY:[0-9]+]] {{[0-9]+}} [[FTY]]
; CHECK-SPIRV: TypeStruct [[TYPETY:[0-9]+]] [[INTTY]] [[FPTRTY]]
; CHECK-SPIRV: TypePointer [[TYPEPTRTY]] {{[0-9]+}} [[TYPETY]]
; CHECK-SPIRV: TypePointer [[DESCPTR1TY:[0-9]+]] {{[0-9]+}} [[TYPETY]]
; CHECK-SPIRV: TypeFunction [[F2TY:[0-9]+]] [[DESCPTRTY]]
; CHECK-SPIRV: TypePointer [[FPTR2TY:[0-9]+]] {{[0-9]+}} [[F2TY]]
; CHECK-SPIRV: TypePointer [[FPTR2ALLOCATY:[0-9]+]] {{[0-9]+}} [[FPTR2TY]]
; CHECK-SPIRV: Variable [[DESCPTR1TY]]
; CHECK-SPIRV: Variable [[FPTR2ALLOCATY]]

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spir64-unknown-unknown"

%struct.Type = type { i32, void (%struct.Desc addrspace(4)*) addrspace(4)* }
%struct.Desc = type { i32, %struct.Type addrspace(4)* }

define spir_kernel void @foo() #0 !kernel_arg_addr_space !4 !kernel_arg_access_qual !4 !kernel_arg_type !4 !kernel_arg_base_type !4 !kernel_arg_type_qual !4 {
entry:
  %t = alloca %struct.Type, align 8
  %ptr = alloca %struct.Desc addrspace(4)* () *, align 8
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

attributes #0 = { "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { inlinehint nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 10.0.0 "}
!4 = !{}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
