; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+all -spirv-text -o %t
; RUN: FileCheck < %t %s

; CHECK: Name [[#NAME:]] "struct._ZTS6Object.Object"
; CHECK-COUNT-1: TypeStruct [[#NAME]]
; TODO add check count one and remove unused, when the type mapping bug is fixed
; CHECK: TypePointer [[#UNUSED:]] {{.*}} [[#NAME]]
; CHECK: TypePointer [[#PTRTY:]] {{.*}} [[#NAME]]
; CHECK-COUNT-2: TypePointer {{.*}} {{.*}} [[#PTRTY]]
; CHECK-NOT: TypePointer {{.*}} {{.*}} [[#UNUSED]]

; ModuleID = 'sycl_test.bc'
source_filename = "sycl_test.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%struct._ZTS4Args.Args = type { %struct._ZTS6Object.Object addrspace(4)* }
%struct._ZTS6Object.Object = type { i32 (%struct._ZTS6Object.Object addrspace(4)*, i32)* }

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func i32 @_Z9somefunc0P4Args(%struct._ZTS4Args.Args addrspace(4)* %args) #0 {
entry:
  %retval = alloca i32, align 4
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %args.addr = alloca %struct._ZTS4Args.Args addrspace(4)*, align 8
  %args.addr.ascast = addrspacecast %struct._ZTS4Args.Args addrspace(4)** %args.addr to %struct._ZTS4Args.Args addrspace(4)* addrspace(4)*
  store %struct._ZTS4Args.Args addrspace(4)* %args, %struct._ZTS4Args.Args addrspace(4)* addrspace(4)* %args.addr.ascast, align 8, !tbaa !5
  ret i32 0
}

; Function Attrs: convergent norecurse nounwind mustprogress
define dso_local spir_func i32 @_Z9somefunc1P6Objecti(%struct._ZTS6Object.Object addrspace(4)* %object, i32 %value) #0 {
entry:
  %retval = alloca i32, align 4
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %object.addr = alloca %struct._ZTS6Object.Object addrspace(4)*, align 8
  %object.addr.ascast = addrspacecast %struct._ZTS6Object.Object addrspace(4)** %object.addr to %struct._ZTS6Object.Object addrspace(4)* addrspace(4)*
  %value.addr = alloca i32, align 4
  %value.addr.ascast = addrspacecast i32* %value.addr to i32 addrspace(4)*
  store %struct._ZTS6Object.Object addrspace(4)* %object, %struct._ZTS6Object.Object addrspace(4)* addrspace(4)* %object.addr.ascast, align 8, !tbaa !5
  store i32 %value, i32 addrspace(4)* %value.addr.ascast, align 4, !tbaa !9
  ret i32 0
}

attributes #0 = { convergent norecurse nounwind mustprogress "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "referenced-indirectly" "stack-protector-buffer-size"="8" "sycl-module-id"="sycl_test.cpp" }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 13.0.0"}
!5 = !{!6, !6, i64 0}
!6 = !{!"any pointer", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10, !10, i64 0}
!10 = !{!"int", !7, i64 0}
