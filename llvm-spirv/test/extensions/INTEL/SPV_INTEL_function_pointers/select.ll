; RUN: llvm-as %s -o %t.bc
; RUN: llvm-spirv %t.bc --spirv-ext=+SPV_INTEL_function_pointers -o %t.spv
; RUN: llvm-spirv %t.spv -to-text -o %t.spt
; RUN: FileCheck < %t.spt %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv -r %t.spv -o %t.r.bc
; RUN: llvm-dis %t.r.bc -o %t.r.ll
; RUN: FileCheck < %t.r.ll %s --check-prefix=CHECK-LLVM

; CHECK-SPIRV-DAG: Name [[#KERNEL_ID:]] "_ZTS6kernel"
; CHECK-SPIRV-DAG: Name [[#BAR:]] "_Z3barii"
; CHECK-SPIRV-DAG: Name [[#BAZ:]] "_Z3bazii"
; CHECK-SPIRV: TypeInt [[#INT32:]] 32
; CHECK-SPIRV: TypeFunction [[#FUNC_TYPE:]] [[#INT32]] [[#INT32]]
; CHECK-SPIRV: TypePointer [[#FUNC_PTR_TYPE:]] [[#]] [[#FUNC_TYPE]]
; CHECK-SPIRV: TypePointer [[#FUNC_PTR_ALLOCA_TYPE:]] [[#]] [[#FUNC_PTR_TYPE]]
; CHECK-SPIRV-DAG: ConstantFunctionPointerINTEL [[#FUNC_PTR_TYPE]] [[#BARPTR:]] [[#BAR]]
; CHECK-SPIRV-DAG: ConstantFunctionPointerINTEL [[#FUNC_PTR_TYPE]] [[#BAZPTR:]] [[#BAZ]]
; CHECK-SPIRV: Function [[#]] [[#KERNEL_ID]]
; CHECK-SPIRV: Variable [[#FUNC_PTR_ALLOCA_TYPE]] [[#FPTR:]]
; CHECK-SPIRV: Select [[#FUNC_PTR_TYPE]] [[#SELECT:]] [[#]] [[#BARPTR]] [[#BAZPTR]]
; CHECK-SPIRV: Store [[#FPTR]] [[#SELECT]]
; CHECK-SPIRV: Load [[#FUNC_PTR_TYPE]] [[#LOAD:]] [[#FPTR]]
; CHECK-SPIRV: FunctionPointerCallINTEL [[#]] [[#]] [[#LOAD]]

; CHECK-LLVM: define spir_kernel void @_ZTS6kernel
; CHECK-LLVM: %[[FPTR_ALLOCA:.*]] = alloca ptr
; CHECK-LLVM: %[[SELECT:.*]] = select i1 %{{.*}}, ptr @_Z3barii, ptr @_Z3bazii
; CHECK-LLVM: store ptr %[[SELECT]], ptr %[[FPTR_ALLOCA]]
; CHECK-LLVM: %[[FPTR:.*]] = load ptr, ptr %[[FPTR_ALLOCA]]
; CHECK-LLVM: call spir_func i32 %[[FPTR]](

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTS6kernel = comdat any

@__spirv_BuiltInGlobalInvocationId = external dso_local local_unnamed_addr addrspace(1) constant <3 x i64>, align 32

; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTS6kernel(ptr addrspace(1) %_arg_, ptr byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_1, ptr byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_2, ptr byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_3) local_unnamed_addr #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %fptr.alloca = alloca ptr, align 8
  %ref.tmp.i = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  %agg.tmp2.i = alloca %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", align 8
  %agg.tmp3.i = alloca %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range", align 8
  %agg.tmp6 = alloca %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", align 8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %agg.tmp2.i)
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %agg.tmp3.i)
  %0 = addrspacecast ptr %agg.tmp2.i to ptr addrspace(4)
  %ptrint4.i = ptrtoint ptr addrspace(4) %0 to i64
  %maskedptr5.i = and i64 %ptrint4.i, 7
  %maskcond6.i = icmp eq i64 %maskedptr5.i, 0
  %1 = addrspacecast ptr %agg.tmp3.i to ptr addrspace(4)
  %ptrint.i = ptrtoint ptr addrspace(4) %1 to i64
  %maskedptr.i = and i64 %ptrint.i, 7
  %maskcond.i = icmp eq i64 %maskedptr.i, 0
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %agg.tmp2.i)
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %agg.tmp3.i)
  %2 = load i64, ptr %_arg_3, align 8
  %add.ptr.i = getelementptr inbounds i32, ptr addrspace(1) %_arg_, i64 %2
  %3 = addrspacecast ptr %agg.tmp6 to ptr addrspace(4)
  %ptrint = ptrtoint ptr addrspace(4) %3 to i64
  %maskedptr = and i64 %ptrint, 7
  %maskcond = icmp eq i64 %maskedptr, 0
  %4 = load <3 x i64>, ptr addrspace(4) addrspacecast (ptr addrspace(1) @__spirv_BuiltInGlobalInvocationId to ptr addrspace(4)), align 32, !noalias !8
  %5 = extractelement <3 x i64> %4, i64 0
  store i64 %5, ptr addrspace(4) %3, align 8, !tbaa !15, !alias.scope !8
  call void @llvm.lifetime.start.p0(i64 8, ptr nonnull %ref.tmp.i) #4
  %6 = addrspacecast ptr %ref.tmp.i to ptr addrspace(4)
  %ptrint.i2 = ptrtoint ptr addrspace(4) %6 to i64
  %maskedptr.i3 = and i64 %ptrint.i2, 7
  %maskcond.i4 = icmp eq i64 %maskedptr.i3, 0
  %rem.i.i = and i64 %5, 1
  %cmp.i.i = icmp eq i64 %rem.i.i, 0
  call void @llvm.lifetime.end.p0(i64 8, ptr nonnull %ref.tmp.i) #4
  %_Z3barii._Z3bazii.i = select i1 %cmp.i.i, ptr @_Z3barii, ptr @_Z3bazii
  store ptr %_Z3barii._Z3bazii.i, ptr %fptr.alloca, align 8
  %fptr = load ptr, ptr %fptr.alloca, align 8
  %call4.i = call spir_func i32 %fptr(i32 10, i32 10), !callees !19
  %arrayidx.i3.i = getelementptr inbounds i32, ptr addrspace(1) %add.ptr.i, i64 %5
  %arrayidx.ascast.i.i = addrspacecast ptr addrspace(1) %arrayidx.i3.i to ptr addrspace(4)
  store i32 %call4.i, ptr addrspace(4) %arrayidx.ascast.i.i, align 4, !tbaa !20
  ret void
}

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.start.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: argmemonly nounwind willreturn
declare void @llvm.lifetime.end.p0(i64 immarg, ptr captures(none)) #1

; Function Attrs: norecurse nounwind readnone
define dso_local spir_func i32 @_Z3barii(i32 %a, i32 %b) local_unnamed_addr #2 {
entry:
  %add = add nsw i32 %b, %a
  ret i32 %add
}

; Function Attrs: norecurse nounwind readnone
define dso_local spir_func i32 @_Z3bazii(i32 %a, i32 %b) local_unnamed_addr #2 {
entry:
  %sub = sub nsw i32 %a, %b
  ret i32 %sub
}

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "sycl-module-id"="f.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #1 = { argmemonly nounwind willreturn }
attributes #2 = { norecurse nounwind readnone "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="false" "stack-protector-buffer-size"="8" "unsafe-fp-math"="false" "use-soft-float"="false" }
attributes #3 = { nounwind willreturn }
attributes #4 = { nounwind }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 11.0.0 "}
!4 = !{i32 1, i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none"}
!6 = !{!"int*", !"cl::sycl::range<1>", !"cl::sycl::range<1>", !"cl::sycl::id<1>"}
!7 = !{!"", !"", !"", !""}
!8 = !{!9, !11, !13}
!9 = distinct !{!9, !10, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv: %agg.result"}
!10 = distinct !{!10, !"_ZN7__spirv29InitSizesSTGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEE8initSizeEv"}
!11 = distinct !{!11, !12, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v: %agg.result"}
!12 = distinct !{!12, !"_ZN7__spirvL22initGlobalInvocationIdILi1EN2cl4sycl2idILi1EEEEET0_v"}
!13 = distinct !{!13, !14, !"_ZN2cl4sycl6detail7Builder5getIdILi1EEEKNS0_2idIXT_EEEv: %agg.result"}
!14 = distinct !{!14, !"_ZN2cl4sycl6detail7Builder5getIdILi1EEEKNS0_2idIXT_EEEv"}
!15 = !{!16, !16, i64 0}
!16 = !{!"long", !17, i64 0}
!17 = !{!"omnipotent char", !18, i64 0}
!18 = !{!"Simple C++ TBAA"}
!19 = !{ptr @_Z3barii, ptr @_Z3bazii}
!20 = !{!21, !21, i64 0}
!21 = !{!"int", !17, i64 0}
