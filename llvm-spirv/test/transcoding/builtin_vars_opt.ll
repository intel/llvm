; RUN: llvm-as < %s -o %t.bc
; RUN: llvm-spirv %t.bc -o %t.spv
; RUN: spirv-val %t.spv
; RUN: llvm-spirv %t.spv -to-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV
; RUN: llvm-spirv %t.spv -r -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-OCL
; RUN: llvm-spirv %t.spv -r --spirv-target-env=SPV-IR -o %t.rev.bc
; RUN: llvm-dis %t.rev.bc -o - | FileCheck %s --check-prefixes=CHECK-LLVM,CHECK-LLVM-SPV

; Check that produced builtin-call-based SPV-IR is recognized by the translator
; RUN: llvm-spirv %t.rev.bc -spirv-text -o - | FileCheck %s --check-prefix=CHECK-SPIRV

; The IR was generated from the following source:
; #include <CL/sycl.hpp>
;
; template <typename T, int N>
; class sycl_subgr;
;
; using namespace cl::sycl;
;
; int main() {
;   queue Queue;
;   int X = 8;
;   nd_range<1> NdRange(X, X);
;   buffer<int> buf(X);
;   Queue.submit([&](handler &cgh) {
;     auto acc = buf.template get_access<access::mode::read_write>(cgh);
;     cgh.parallel_for<sycl_subgr<int, 0>>(NdRange, [=](nd_item<1> NdItem) {
;       intel::sub_group SG = NdItem.get_sub_group();
;       if (X % 2) {
;         acc[0] = SG.get_max_local_range()[0];
;       }
;       acc[1] = (X % 3) ? 1 : SG.get_max_local_range()[0];
;     });
;   });
;   return 0;
; }
; Command line:
; clang -fsycl -fsycl-device-only -Xclang -fsycl-enable-optimizations tmp.cpp -o tmp.bc
; llvm-spirv tmp.bc -s -o builtin_vars_opt.ll

; CHECK-SPIRV: Decorate [[#SG_MaxSize_BI:]] BuiltIn 37
; CHECK-SPIRV: Decorate [[#SG_MaxSize_BI:]] Constant
; CHECK-SPIRV: Decorate [[#SG_MaxSize_BI:]] LinkageAttributes "__spirv_BuiltInSubgroupMaxSize" Import
;
; CHECK-LLVM-OCL-NOT: @__spirv_BuiltInSubgroupMaxSize
; CHECK-LLVM-NOT: addrspacecast i32 addrspace(1)* @__spirv_BuiltInSubgroupMaxSize to i32 addrspace(4)*
; CHECK-LLVM-LABEL: if.then.i
; CHECK-LLVM-NOT: load
; CHECK-LLVM-OCL: call spir_func i32 @_Z22get_max_sub_group_sizev()
; CHECK-LLVM-SPV: call spir_func i32 @_Z30__spirv_BuiltInSubgroupMaxSizev()
; CHECK-LLVM-LABEL: cond.false.i:
; CHECK-LLVM-NOT: load
; CHECK-LLVM-OCL: call spir_func i32 @_Z22get_max_sub_group_sizev()
; CHECK-LLVM-SPV: call spir_func i32 @_Z30__spirv_BuiltInSubgroupMaxSizev()

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-linux"

%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }
%"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" = type { [1 x i64] }
%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id" = type { %"class._ZTSN2cl4sycl6detail5arrayILi1EEE.cl::sycl::detail::array" }

$_ZTS10sycl_subgrIiLi0EE = comdat any

@__spirv_BuiltInSubgroupMaxSize = external dso_local local_unnamed_addr addrspace(1) constant i32, align 4


; Function Attrs: norecurse
define weak_odr dso_local spir_kernel void @_ZTS10sycl_subgrIiLi0EE(i32 %_arg_, i32 addrspace(1)* %_arg_1, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_3, %"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range"* byval(%"class._ZTSN2cl4sycl5rangeILi1EEE.cl::sycl::range") align 8 %_arg_4, %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* byval(%"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id") align 8 %_arg_5) local_unnamed_addr #0 comdat !kernel_arg_addr_space !4 !kernel_arg_access_qual !5 !kernel_arg_type !6 !kernel_arg_base_type !6 !kernel_arg_type_qual !7 {
entry:
  %0 = getelementptr inbounds %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id", %"class._ZTSN2cl4sycl2idILi1EEE.cl::sycl::id"* %_arg_5, i64 0, i32 0, i32 0, i64 0
  %1 = load i64, i64* %0, align 8
  %add.ptr.i = getelementptr inbounds i32, i32 addrspace(1)* %_arg_1, i64 %1
  %2 = and i32 %_arg_, 1
  %tobool.not.i = icmp eq i32 %2, 0
  %3 = addrspacecast i32 addrspace(1)* @__spirv_BuiltInSubgroupMaxSize to i32 addrspace(4)*
  br i1 %tobool.not.i, label %if.end.i, label %if.then.i

if.then.i:                                        ; preds = %entry
  %4 = load i32, i32 addrspace(4)* %3, align 4, !noalias !8
  %ptridx.ascast.i14.i = addrspacecast i32 addrspace(1)* %add.ptr.i to i32 addrspace(4)*
  store i32 %4, i32 addrspace(4)* %ptridx.ascast.i14.i, align 4
  br label %if.end.i

if.end.i:                                         ; preds = %if.then.i, %entry
  %rem3.i = srem i32 %_arg_, 3
  %tobool4.not.i = icmp eq i32 %rem3.i, 0
  br i1 %tobool4.not.i, label %cond.false.i, label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_7nd_itemILi1EEEE_clES5_.exit"

cond.false.i:                                     ; preds = %if.end.i
  %5 = load i32, i32 addrspace(4)* %3, align 4, !noalias !11
  br label %"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_7nd_itemILi1EEEE_clES5_.exit"

"_ZZZ4mainENK3$_0clERN2cl4sycl7handlerEENKUlNS1_7nd_itemILi1EEEE_clES5_.exit": ; preds = %cond.false.i, %if.end.i
  %cond.i = phi i32 [ %5, %cond.false.i ], [ 1, %if.end.i ]
  %ptridx.i.i = getelementptr inbounds i32, i32 addrspace(1)* %add.ptr.i, i64 1
  %ptridx.ascast.i.i = addrspacecast i32 addrspace(1)* %ptridx.i.i to i32 addrspace(4)*
  store i32 %cond.i, i32 addrspace(4)* %ptridx.ascast.i.i, align 4
  ret void
}

attributes #0 = { norecurse "correctly-rounded-divide-sqrt-fp-math"="false" "disable-tail-calls"="false" "frame-pointer"="all" "less-precise-fpmad"="false" "min-legal-vector-width"="0" "no-infs-fp-math"="false" "no-jump-tables"="false" "no-nans-fp-math"="false" "no-signed-zeros-fp-math"="false" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="../sycl/test/sub_group/shuffle.cpp" "uniform-work-group-size"="true" "unsafe-fp-math"="false" "use-soft-float"="false" }

!llvm.module.flags = !{!0}
!opencl.spir.version = !{!1}
!spirv.Source = !{!2}
!llvm.ident = !{!3}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 1, i32 2}
!2 = !{i32 4, i32 100000}
!3 = !{!"clang version 12.0.0 (https://github.com/intel/llvm.git e5ea144e480c7b28162a1477b3d462cfc221ff61)"}
!4 = !{i32 0, i32 1, i32 0, i32 0, i32 0}
!5 = !{!"none", !"none", !"none", !"none", !"none"}
!6 = !{!"int", !"int*", !"cl::sycl::range<1>", !"cl::sycl::range<1>", !"cl::sycl::id<1>"}
!7 = !{!"", !"", !"", !"", !""}
!8 = !{!9, !9, i64 0}
!9 = !{!"int", !10, i64 0}
!10 = !{!"omnipotent char", !11, i64 0}
!11 = !{!"Simple C++ TBAA"}
!12 = !{!13}
!13 = distinct !{!13, !14, !"_ZNK2cl4sycl5intel9sub_group19get_max_local_rangeEv: %agg.result"}
!14 = distinct !{!14, !"_ZNK2cl4sycl5intel9sub_group19get_max_local_rangeEv"}
!15 = !{!16}
!16 = distinct !{!16, !17, !"_ZNK2cl4sycl5intel9sub_group19get_max_local_rangeEv: %agg.result"}
!17 = distinct !{!17, !"_ZNK2cl4sycl5intel9sub_group19get_max_local_rangeEv"}
