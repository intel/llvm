; This test checks whether subroutine arguments are converted
; correctly to llvm's native vector type.
;
; RUN: opt < %s -ESIMDLowerVecArg -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%class._ZTS4simdIiLi16EE.simd = type { <16 x i32> }

$_ZN4simdIiLi16EEC1ERS0_ = comdat any

$_ZN4simdIiLi16EEC2ERS0_ = comdat any

; Function Attrs: norecurse nounwind
define spir_func void @_Z3fooi(i32 %x) #0 {
entry:
  %x.addr = alloca i32, align 4
; CHECK: {{.+}} = alloca {{.+}}
; CHECK-NEXT: [[A:%[a-zA-Z0-9_]*]] = alloca {{.+}}
  %a = alloca %class._ZTS4simdIiLi16EE.simd, align 64
  %agg.tmp = alloca %class._ZTS4simdIiLi16EE.simd, align 64
  store i32 %x, i32* %x.addr, align 4, !tbaa !4
  %0 = bitcast %class._ZTS4simdIiLi16EE.simd* %a to i8*
  call void @llvm.lifetime.start.p0i8(i64 64, i8* %0) #2
; CHECK: [[ADDRSPCAST1:%[a-zA-Z0-9_]*]] = addrspacecast {{.+}} [[A]] to {{.+}}
  %1 = addrspacecast %class._ZTS4simdIiLi16EE.simd* %agg.tmp to %class._ZTS4simdIiLi16EE.simd addrspace(4)*
  %2 = addrspacecast %class._ZTS4simdIiLi16EE.simd* %a to %class._ZTS4simdIiLi16EE.simd addrspace(4)*
; CHECK: [[BITCASTRESULT1:%[a-zA-Z0-9_]*]] = bitcast {{.+}} addrspace(4)* [[ADDRSPCAST1]] to <16 x i32> addrspace(4)*
; CHECK-NEXT: call spir_func void @_ZN4simdIiLi16EEC1ERS0_(<16 x i32> addrspace(4)* {{.+}}, <16 x i32> addrspace(4)* [[BITCASTRESULT1]])
  call spir_func void @_ZN4simdIiLi16EEC1ERS0_(%class._ZTS4simdIiLi16EE.simd addrspace(4)* %1, %class._ZTS4simdIiLi16EE.simd addrspace(4)* align 64 dereferenceable(64) %2)
; CHECK: [[BITCASTRESULT2:%[a-zA-Z0-9_]*]] = bitcast {{.+}} to <16 x i32>*
; CHECK-NEXT: {{.+}} = call spir_func i32 {{.+}}bar{{.+}}(<16 x i32>* [[BITCASTRESULT2]])
  %call = call spir_func i32 @_Z3bar4simdIiLi16EE(%class._ZTS4simdIiLi16EE.simd* %agg.tmp)
  %3 = bitcast %class._ZTS4simdIiLi16EE.simd* %a to i8*
  call void @llvm.lifetime.end.p0i8(i64 64, i8* %3) #2
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind
; CHECK: define spir_func i32 @_Z3bar4simdIiLi16EE(<16 x i32>* {{.+}}
define spir_func i32 @_Z3bar4simdIiLi16EE(%class._ZTS4simdIiLi16EE.simd* %v) #0 {
entry:
; CHECK: {{.+}} = bitcast <16 x i32>* {{.+}}
  ret i32 1
}

; Function Attrs: norecurse nounwind
; CHECK: define linkonce_odr spir_func void @_ZN4simdIiLi16EEC1ERS0_(<16 x i32> addrspace(4)* [[OLDARG0:%[a-zA-Z0-9_]*]], <16 x i32> addrspace(4)*{{.*}} [[OLDARG1:%[a-zA-Z0-9_]*]]) unnamed_addr {{.+}}
define linkonce_odr spir_func void @_ZN4simdIiLi16EEC1ERS0_(%class._ZTS4simdIiLi16EE.simd addrspace(4)* %this, %class._ZTS4simdIiLi16EE.simd addrspace(4)* align 64 dereferenceable(64) %other) unnamed_addr #0 comdat align 2 {
entry:
; CHECK: [[NEWARG1:%[a-zA-Z0-9_]*]] = bitcast <16 x i32> addrspace(4)* [[OLDARG1]] to {{.+}}
; CHECK-NEXT: [[NEWARG0:%[a-zA-Z0-9_]*]] = bitcast <16 x i32> addrspace(4)* [[OLDARG0]] to {{.+}}
  %this.addr = alloca %class._ZTS4simdIiLi16EE.simd addrspace(4)*, align 8
  %other.addr = alloca %class._ZTS4simdIiLi16EE.simd addrspace(4)*, align 8
; CHECK: store {{.+}} addrspace(4)* [[NEWARG0]], {{.+}}
  store %class._ZTS4simdIiLi16EE.simd addrspace(4)* %this, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %this.addr, align 8, !tbaa !8
; CHECK-NEXT: store {{.+}} addrspace(4)* [[NEWARG1]], {{.+}}
  store %class._ZTS4simdIiLi16EE.simd addrspace(4)* %other, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %other.addr, align 8, !tbaa !8
  %this1 = load %class._ZTS4simdIiLi16EE.simd addrspace(4)*, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %this.addr, align 8
  %0 = load %class._ZTS4simdIiLi16EE.simd addrspace(4)*, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %other.addr, align 8
  call spir_func void @_ZN4simdIiLi16EEC2ERS0_(%class._ZTS4simdIiLi16EE.simd addrspace(4)* %this1, %class._ZTS4simdIiLi16EE.simd addrspace(4)* align 64 dereferenceable(64) %0)
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0i8(i64 immarg, i8* nocapture) #1

; Function Attrs: norecurse nounwind
define linkonce_odr spir_func void @_ZN4simdIiLi16EEC2ERS0_(%class._ZTS4simdIiLi16EE.simd addrspace(4)* %this, %class._ZTS4simdIiLi16EE.simd addrspace(4)* align 64 dereferenceable(64) %other) unnamed_addr #0 comdat align 2 {
entry:
  %this.addr = alloca %class._ZTS4simdIiLi16EE.simd addrspace(4)*, align 8
  %other.addr = alloca %class._ZTS4simdIiLi16EE.simd addrspace(4)*, align 8
  store %class._ZTS4simdIiLi16EE.simd addrspace(4)* %this, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %this.addr, align 8, !tbaa !8
  store %class._ZTS4simdIiLi16EE.simd addrspace(4)* %other, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %other.addr, align 8, !tbaa !8
  %this1 = load %class._ZTS4simdIiLi16EE.simd addrspace(4)*, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %this.addr, align 8
  %0 = load %class._ZTS4simdIiLi16EE.simd addrspace(4)*, %class._ZTS4simdIiLi16EE.simd addrspace(4)** %other.addr, align 8, !tbaa !8
  %__M_data = getelementptr inbounds %class._ZTS4simdIiLi16EE.simd, %class._ZTS4simdIiLi16EE.simd addrspace(4)* %0, i32 0, i32 0
  %1 = load <16 x i32>, <16 x i32> addrspace(4)* %__M_data, align 64, !tbaa !10
  %__M_data2 = getelementptr inbounds %class._ZTS4simdIiLi16EE.simd, %class._ZTS4simdIiLi16EE.simd addrspace(4)* %this1, i32 0, i32 0
  store <16 x i32> %1, <16 x i32> addrspace(4)* %__M_data2, align 64, !tbaa !10
  ret void
}

attributes #0 = { norecurse nounwind "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="none" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0 (https://github.com/kbobrovs/llvm.git fb752d6351dc6785f5438b137a86fa39a3493225)"}
!4 = !{!5, !5, i64 0}
!5 = !{!"int", !6, i64 0}
!6 = !{!"omnipotent char", !7, i64 0}
!7 = !{!"Simple C++ TBAA"}
!8 = !{!9, !9, i64 0}
!9 = !{!"any pointer", !6, i64 0}
!10 = !{!6, !6, i64 0}
