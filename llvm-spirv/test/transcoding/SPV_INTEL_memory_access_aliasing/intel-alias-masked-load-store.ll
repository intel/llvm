;; Compiled from:
;; template <typename name, typename Func>
;; __attribute__((sycl_kernel)) void kernel(const Func &kernelFunc) {
;;   kernelFunc();
;; }
;;
;; int main() {
;;   int *a;
;;   int *b;
;;   int *c;
;;   kernel<class kernel_restrict>(
;;       [ a, b, c ]() [[intel::kernel_args_restrict]] { c[0] = a[0] + b[0]; });
;;
;;   int *d;
;;   int *e;
;;   int *f;
;;
;;   int g = 42;
;;   kernel<class kernel_restrict_other_types>(
;;       [ a, b, c, g ]() [[intel::kernel_args_restrict]] { c[0] = a[0] + b[0] + g; });
;;
;; with:
;; clang++ -fsycl -fsycl-is-device %s -o -
;; using https://github.com/intel/llvm.git bb5a2fece7c3315d7c72d8495c34a8a6eabc92d8
;; where generated loads and stores where replaced with calls to @wrappedload
;; and @wrappedstore functions

; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_memory_access_aliasing -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.bc -o %t.n.spv
; RUN: llvm-spirv %t.n.spv -to-text -o %t.n.spt
; RUN: FileCheck < %t.n.spt %s --check-prefix=CHECK-SPIRV-NEGATIVE
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV: Capability MemoryAccessAliasingINTEL
; CHECK-SPIRV: Extension "SPV_INTEL_memory_access_aliasing"
; CHECK-SPIRV: AliasDomainDeclINTEL [[DOMAIN1:[0-9]+]]
; CHECK-SPIRV: AliasScopeDeclINTEL [[SCOPE1:[0-9]+]] [[DOMAIN1]]
; CHECK-SPIRV: AliasScopeListDeclINTEL [[LIST1:[0-9]+]] [[SCOPE1]]
; CHECK-SPIRV: AliasDomainDeclINTEL [[DOMAIN2:[0-9]+]]
; CHECK-SPIRV: AliasScopeDeclINTEL [[SCOPE2:[0-9]+]] [[DOMAIN2]]
; CHECK-SPIRV: AliasScopeListDeclINTEL [[LIST2:[0-9]+]] [[SCOPE2]]
; CHECK-SPIRV: AliasDomainDeclINTEL [[DOMAIN3:[0-9]+]]
; CHECK-SPIRV: AliasScopeDeclINTEL [[SCOPE3:[0-9]+]] [[DOMAIN3]]
; CHECK-SPIRV: AliasScopeListDeclINTEL [[LIST3:[0-9]+]] [[SCOPE3]]
; CHECK-SPIRV: DecorateId [[TARGET1:[0-9]+]] AliasScopeINTEL [[LIST1]]
; CHECK-SPIRV: DecorateId [[TARGET2:[0-9]+]] AliasScopeINTEL [[LIST1]]
; CHECK-SPIRV: DecorateId [[TARGET3:[0-9]+]] AliasScopeINTEL [[LIST3]]
; CHECK-SPIRV: DecorateId [[TARGET4:[0-9]+]] AliasScopeINTEL [[LIST3]]
; CHECK-SPIRV: DecorateId [[TARGET5:[0-9]+]] NoAliasINTEL [[LIST1]]
; CHECK-SPIRV: DecorateId [[TARGET2]] NoAliasINTEL [[LIST2]]
; CHECK-SPIRV: DecorateId [[TARGET6:[0-9]+]] NoAliasINTEL [[LIST3]]
; CHECK-SPIRV: FunctionCall {{.*}} [[TARGET1]]
; CHECK-SPIRV: FunctionCall {{.*}} [[TARGET2]]
; CHECK-SPIRV: FunctionCall {{.*}} [[TARGET5]]
; CHECK-SPIRV: FunctionCall {{.*}} [[TARGET3]]
; CHECK-SPIRV: FunctionCall {{.*}} [[TARGET4]]
; CHECK-SPIRV: FunctionCall {{.*}} [[TARGET6]]

; CHECK-SPIRV-NEGATIVE-NOT: Capability MemoryAccessAliasingINTEL
; CHECK-SPIRV-NEGATIVE-NOT: Extension "SPV_INTEL_memory_access_aliasing"
; CHECK-SPIRV-NEGATIVE-NOT: AliasDomainDeclINTEL
; CHECK-SPIRV-NEGATIVE-NOT: AliasScopeDeclINTEL
; CHECK-SPIRV-NEGATIVE-NOT: AliasScopeListDeclINTEL
; CHECK-SPIRV-NEGATIVE-NOT: DecorateId {{.*}} AliasScopeINTEL
; CHECK-SPIRV-NEGATIVE-NOT: DecorateId {{.*}} NoAliasINTEL

; ModuleID = 'optimized_intel_restrict.bc'
source_filename = "intel_restrict.cpp"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: nofree norecurse nounwind willreturn mustprogress
define dso_local spir_kernel void @_ZTSZ4mainE15kernel_restrict(i32 addrspace(1)* noalias %_arg_, i32 addrspace(1)* noalias %_arg_1, i32 addrspace(1)* noalias %_arg_3) local_unnamed_addr #0 !kernel_arg_buffer_location !4 {
entry:
  %0 = addrspacecast i32 addrspace(1)* %_arg_ to i32 addrspace(4)*
  %1 = addrspacecast i32 addrspace(1)* %_arg_1 to i32 addrspace(4)*
  %2 = addrspacecast i32 addrspace(1)* %_arg_3 to i32 addrspace(4)*
; CHECK-LLVM: call spir_func i32 @wrappedload{{.*}} !alias.scope ![[LISTMD1:[0-9]+]]
; CHECK-LLVM: call spir_func i32 @wrappedload{{.*}} !alias.scope ![[LISTMD1]]{{.*}} ![[LISTMD2:[0-9]+]]
  %3 = call i32 @wrappedload(i32 addrspace(4)* %0), !tbaa !5, !alias.scope !9
  %4 = call i32 @wrappedload(i32 addrspace(4)* %1), !tbaa !5, !alias.scope !9, !noalias !16
  %add.i = add nsw i32 %4, %3
; CHECK-LLVM: call spir_func void @wrappedstore{{.*}} !noalias ![[LISTMD1]]
  call void @wrappedstore(i32 %add.i, i32 addrspace(4)* %2),!tbaa !5, !noalias !9
  ret void
}

; Function Attrs: nofree norecurse nounwind willreturn mustprogress
define dso_local spir_kernel void @_ZTSZ4mainE27kernel_restrict_other_types(i32 addrspace(1)* noalias %_arg_, i32 addrspace(1)* noalias %_arg_1, i32 addrspace(1)* noalias %_arg_3, i32 %_arg_5) local_unnamed_addr #0 !kernel_arg_buffer_location !12 {
entry:
  %0 = addrspacecast i32 addrspace(1)* %_arg_ to i32 addrspace(4)*
  %1 = addrspacecast i32 addrspace(1)* %_arg_1 to i32 addrspace(4)*
  %2 = addrspacecast i32 addrspace(1)* %_arg_3 to i32 addrspace(4)*
; CHECK-LLVM: call spir_func i32 @wrappedload{{.*}} !alias.scope ![[LISTMD3:[0-9]+]]
; CHECK-LLVM: call spir_func i32 @wrappedload{{.*}} !alias.scope ![[LISTMD3]]
  %3 = call i32 @wrappedload(i32 addrspace(4)* %0), !tbaa !5, !alias.scope !13
  %4 = call i32 @wrappedload(i32 addrspace(4)* %1), !tbaa !5, !alias.scope !13
  %add.i = add i32 %3, %_arg_5
  %add3.i = add i32 %add.i, %4
; CHECK-LLVM: call spir_func void @wrappedstore{{.*}} !noalias ![[LISTMD3]]
  call void @wrappedstore(i32 %add3.i, i32 addrspace(4)* %2),!tbaa !5, !noalias !13
  ret void
}

; Function Attrs: norecurse nounwind readnone
define dso_local spir_func i32 @wrappedload(i32 addrspace(4)* %0) local_unnamed_addr #2 {
entry:
  %1 = load i32, i32 addrspace(4)* %0, align 4
  ret i32 %1
}

; Function Attrs: norecurse nounwind readnone
define dso_local spir_func void @wrappedstore(i32 %0, i32 addrspace(4)* %1) local_unnamed_addr #2 {
entry:
  store i32 %0, i32 addrspace(4)* %1, align 4
  ret void
}

attributes #0 = { nofree norecurse nounwind willreturn mustprogress "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="intel-restrict.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 13.0.0 (https://github.com/intel/llvm.git bb5a2fece7c3315d7c72d8495c34a8a6eabc92d8)"}
!4 = !{i32 -1, i32 -1, i32 -1}
!5 = !{!6, !6, i64 0}
!6 = !{!"int", !7, i64 0}
!7 = !{!"omnipotent char", !8, i64 0}
!8 = !{!"Simple C++ TBAA"}
!9 = !{!10}
!10 = distinct !{!10, !11, !"_ZZ4mainENK3$_0clEv: %this"}
!11 = distinct !{!11, !"_ZZ4mainENK3$_0clEv"}
!12 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!13 = !{!14}
!14 = distinct !{!14, !15, !"_ZZ4mainENK3$_2clEv: %this"}
!15 = distinct !{!15, !"_ZZ4mainENK3$_2clEv"}
!16 = !{!17}
!17 = distinct !{!17, !18, !"_ZZ4mainENK3$_0clEv: %this"}
!18 = distinct !{!18, !"_ZZ4mainENK3$_0clEv"}

; CHECK-LLVM: ![[LISTMD1]] = !{![[SCOPEMD1:[0-9]+]]}
; CHECK-LLVM: ![[SCOPEMD1]] = distinct !{![[SCOPEMD1]], ![[DOMAINMD1:[0-9]+]]}
; CHECK-LLVM: ![[DOMAINMD1]] = distinct !{![[DOMAINMD1]]}
; CHECK-LLVM: ![[LISTMD2]] = !{![[SCOPEMD2:[0-9]+]]}
; CHECK-LLVM: ![[SCOPEMD2]] = distinct !{![[SCOPEMD2]], ![[DOMAINMD2:[0-9]+]]}
; CHECK-LLVM: ![[DOMAINMD2]] = distinct !{![[DOMAINMD2]]}
; CHECK-LLVM: ![[LISTMD3]] = !{![[SCOPEMD3:[0-9]+]]}
; CHECK-LLVM: ![[SCOPEMD3]] = distinct !{![[SCOPEMD3]], ![[DOMAINMD3:[0-9]+]]}
; CHECK-LLVM: ![[DOMAINMD3]] = distinct !{![[DOMAINMD3]]}
