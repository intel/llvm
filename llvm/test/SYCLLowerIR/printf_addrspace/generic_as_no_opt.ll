;; This tests replacement of string literal address space for __spirv_ocl_printf
;; when no optimizations (inlining, constant propagation) have been performed prior
;; to the pass scheduling.

;; Compiled with the following command (custom build of SYCL Clang with
;; SYCLMutatePrintfAddrspacePass turned off):
;; clang++ -fsycl -fsycl-device-only Inputs/experimental-printf.cpp -S -O0 -D__SYCL_USE_NON_VARIADIC_SPIRV_OCL_PRINTF__

; RUN: opt < %s --SYCLMutatePrintfAddrspace -S | FileCheck %s
; RUN: opt < %s --SYCLMutatePrintfAddrspace -S --enable-new-pm=1 | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

%"class.cl::sycl::range" = type { %"class.cl::sycl::detail::array" }
%"class.cl::sycl::detail::array" = type { [1 x i64] }
%"class.cl::sycl::id" = type { %"class.cl::sycl::detail::array" }
%class.anon = type { %"class.cl::sycl::accessor" }
%"class.cl::sycl::accessor" = type { %"class.cl::sycl::detail::AccessorImplDevice" }
%"class.cl::sycl::detail::AccessorImplDevice" = type { %"class.cl::sycl::id", %"class.cl::sycl::range", %"class.cl::sycl::range" }
%"class.cl::sycl::detail::accessor_common" = type { i8 }
%class.anon.0 = type { i8 }

$_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev = comdat any

$_ZN2cl4sycl2idILi1EEC2Ev = comdat any

$_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv = comdat any

$_ZN2cl4sycl6detail18AccessorImplDeviceILi1EEC2ENS0_2idILi1EEENS0_5rangeILi1EEES7_ = comdat any

$_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE = comdat any

$_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE = comdat any

$_ZN2cl4sycl6detail5arrayILi1EEixEi = comdat any

$_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE9getOffsetEv = comdat any

$_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getAccessRangeEv = comdat any

$_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getMemoryRangeEv = comdat any

$_ZNK2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEixILi1EvEERS3_NS0_2idILi1EEE = comdat any

$_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE = comdat any

$_ZNK2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEE = comdat any

$_ZNK2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE15getQualifiedPtrEv = comdat any

$_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_ = comdat any

$_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_ = comdat any

$_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_ = comdat any

; CHECK-DAG: @.str._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %f\0A\00", align 1
@.str = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %f\0A\00", align 1
; CHECK-DAG: @.str.1._AS2 = internal addrspace(2) constant [15 x i8] c"String No. %i\0A\00", align 1
@.str.1 = private unnamed_addr addrspace(1) constant [15 x i8] c"String No. %i\0A\00", align 1

; Function Attrs: convergent noinline norecurse optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEEC2Ev(%"class.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(32) %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::accessor" addrspace(4)*, align 8
  %agg.tmp = alloca %"class.cl::sycl::id", align 8
  %agg.tmp2 = alloca %"class.cl::sycl::range", align 8
  %agg.tmp3 = alloca %"class.cl::sycl::range", align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::accessor" addrspace(4)** %this.addr to %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %agg.tmp.ascast = addrspacecast %"class.cl::sycl::id"* %agg.tmp to %"class.cl::sycl::id" addrspace(4)*
  %agg.tmp2.ascast = addrspacecast %"class.cl::sycl::range"* %agg.tmp2 to %"class.cl::sycl::range" addrspace(4)*
  %agg.tmp3.ascast = addrspacecast %"class.cl::sycl::range"* %agg.tmp3 to %"class.cl::sycl::range" addrspace(4)*
  store %"class.cl::sycl::accessor" addrspace(4)* %this, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::accessor" addrspace(4)*, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class.cl::sycl::accessor" addrspace(4)* %this1 to %"class.cl::sycl::detail::accessor_common" addrspace(4)*
  %impl = getelementptr inbounds %"class.cl::sycl::accessor", %"class.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %1 = bitcast %"class.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to i8 addrspace(4)*
  call void @llvm.memset.p4i8.i64(i8 addrspace(4)* align 8 %1, i8 0, i64 8, i1 false)
  call spir_func void @_ZN2cl4sycl2idILi1EEC2Ev(%"class.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(8) %agg.tmp.ascast) #8
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv(%"class.cl::sycl::range" addrspace(4)* sret(%"class.cl::sycl::range") align 8 %agg.tmp2.ascast) #8
  call spir_func void @_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv(%"class.cl::sycl::range" addrspace(4)* sret(%"class.cl::sycl::range") align 8 %agg.tmp3.ascast) #8
  %agg.tmp.ascast.ascast = addrspacecast %"class.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to %"class.cl::sycl::id"*
  %agg.tmp2.ascast.ascast = addrspacecast %"class.cl::sycl::range" addrspace(4)* %agg.tmp2.ascast to %"class.cl::sycl::range"*
  %agg.tmp3.ascast.ascast = addrspacecast %"class.cl::sycl::range" addrspace(4)* %agg.tmp3.ascast to %"class.cl::sycl::range"*
  call spir_func void @_ZN2cl4sycl6detail18AccessorImplDeviceILi1EEC2ENS0_2idILi1EEENS0_5rangeILi1EEES7_(%"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* align 8 dereferenceable_or_null(24) %impl, %"class.cl::sycl::id"* byval(%"class.cl::sycl::id") align 8 %agg.tmp.ascast.ascast, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %agg.tmp2.ascast.ascast, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %agg.tmp3.ascast.ascast) #8
  ret void
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* noalias nocapture writeonly, i8 addrspace(4)* noalias nocapture readonly, i64, i1 immarg) #3

; Function Attrs: convergent mustprogress noinline norecurse optnone
define internal spir_func void @_ZZZN2cl4sycl6detailL19submitAssertCaptureERNS0_5queueERNS0_5eventEPS2_RKNS1_13code_locationEENKUlRNS0_7handlerEE_clESB_ENKUlvE_clEv(%class.anon addrspace(4)* align 8 dereferenceable_or_null(32) %this) #2 align 2 {
entry:
  %this.addr = alloca %class.anon addrspace(4)*, align 8
  %agg.tmp = alloca %"class.cl::sycl::id", align 8
  %this.addr.ascast = addrspacecast %class.anon addrspace(4)** %this.addr to %class.anon addrspace(4)* addrspace(4)*
  %agg.tmp.ascast = addrspacecast %"class.cl::sycl::id"* %agg.tmp to %"class.cl::sycl::id" addrspace(4)*
  store %class.anon addrspace(4)* %this, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon addrspace(4)*, %class.anon addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = getelementptr inbounds %class.anon, %class.anon addrspace(4)* %this1, i32 0, i32 0
  call spir_func void @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(8) %agg.tmp.ascast, i64 0) #8
  %agg.tmp.ascast.ascast = addrspacecast %"class.cl::sycl::id" addrspace(4)* %agg.tmp.ascast to %"class.cl::sycl::id"*
  ret void
}

; Function Attrs: argmemonly nofree nounwind willreturn writeonly
declare void @llvm.memset.p4i8.i64(i8 addrspace(4)* nocapture writeonly, i8, i64, i1 immarg) #4

; Function Attrs: convergent noinline norecurse optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl2idILi1EEC2Ev(%"class.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(8) %this) unnamed_addr #1 comdat align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::id" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::id" addrspace(4)** %this.addr to %"class.cl::sycl::id" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::id" addrspace(4)* %this, %"class.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::id" addrspace(4)*, %"class.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class.cl::sycl::id" addrspace(4)* %this1 to %"class.cl::sycl::detail::array" addrspace(4)*
  call spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(8) %0, i64 0) #8
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail14InitializedValILi1ENS0_5rangeEE3getILi0EEENS3_ILi1EEEv(%"class.cl::sycl::range" addrspace(4)* noalias sret(%"class.cl::sycl::range") align 8 %agg.result) #2 comdat align 2 {
entry:
  call spir_func void @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::range" addrspace(4)* align 8 dereferenceable_or_null(8) %agg.result, i64 0) #8
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail18AccessorImplDeviceILi1EEC2ENS0_2idILi1EEENS0_5rangeILi1EEES7_(%"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* align 8 dereferenceable_or_null(24) %this, %"class.cl::sycl::id"* byval(%"class.cl::sycl::id") align 8 %Offset, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %AccessRange, %"class.cl::sycl::range"* byval(%"class.cl::sycl::range") align 8 %MemoryRange) unnamed_addr #5 comdat align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)** %this.addr to %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this, %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %Offset.ascast = addrspacecast %"class.cl::sycl::id"* %Offset to %"class.cl::sycl::id" addrspace(4)*
  %AccessRange.ascast = addrspacecast %"class.cl::sycl::range"* %AccessRange to %"class.cl::sycl::range" addrspace(4)*
  %MemoryRange.ascast = addrspacecast %"class.cl::sycl::range"* %MemoryRange to %"class.cl::sycl::range" addrspace(4)*
  %this1 = load %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)*, %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %Offset2 = getelementptr inbounds %"class.cl::sycl::detail::AccessorImplDevice", %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 0
  %0 = bitcast %"class.cl::sycl::id" addrspace(4)* %Offset2 to i8 addrspace(4)*
  %1 = bitcast %"class.cl::sycl::id" addrspace(4)* %Offset.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %0, i8 addrspace(4)* align 8 %1, i64 8, i1 false)
  %AccessRange3 = getelementptr inbounds %"class.cl::sycl::detail::AccessorImplDevice", %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 1
  %2 = bitcast %"class.cl::sycl::range" addrspace(4)* %AccessRange3 to i8 addrspace(4)*
  %3 = bitcast %"class.cl::sycl::range" addrspace(4)* %AccessRange.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %2, i8 addrspace(4)* align 8 %3, i64 8, i1 false)
  %MemRange = getelementptr inbounds %"class.cl::sycl::detail::AccessorImplDevice", %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %this1, i32 0, i32 2
  %4 = bitcast %"class.cl::sycl::range" addrspace(4)* %MemRange to i8 addrspace(4)*
  %5 = bitcast %"class.cl::sycl::range" addrspace(4)* %MemoryRange.ascast to i8 addrspace(4)*
  call void @llvm.memcpy.p4i8.p4i8.i64(i8 addrspace(4)* align 8 %4, i8 addrspace(4)* align 8 %5, i64 8, i1 false)
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(8) %this, i64 %dim0) unnamed_addr #5 comdat align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::detail::array" addrspace(4)*, align 8
  %dim0.addr = alloca i64, align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::detail::array" addrspace(4)** %this.addr to %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dim0.addr.ascast = addrspacecast i64* %dim0.addr to i64 addrspace(4)*
  store %"class.cl::sycl::detail::array" addrspace(4)* %this, %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  store i64 %dim0, i64 addrspace(4)* %dim0.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::detail::array" addrspace(4)*, %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %common_array = getelementptr inbounds %"class.cl::sycl::detail::array", %"class.cl::sycl::detail::array" addrspace(4)* %this1, i32 0, i32 0
  %arrayinit.begin = getelementptr inbounds [1 x i64], [1 x i64] addrspace(4)* %common_array, i64 0, i64 0
  %0 = load i64, i64 addrspace(4)* %dim0.addr.ascast, align 8
  store i64 %0, i64 addrspace(4)* %arrayinit.begin, align 8
  ret void
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl5rangeILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::range" addrspace(4)* align 8 dereferenceable_or_null(8) %this, i64 %dim0) unnamed_addr #5 comdat align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::range" addrspace(4)*, align 8
  %dim0.addr = alloca i64, align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::range" addrspace(4)** %this.addr to %"class.cl::sycl::range" addrspace(4)* addrspace(4)*
  %dim0.addr.ascast = addrspacecast i64* %dim0.addr to i64 addrspace(4)*
  store %"class.cl::sycl::range" addrspace(4)* %this, %"class.cl::sycl::range" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  store i64 %dim0, i64 addrspace(4)* %dim0.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::range" addrspace(4)*, %"class.cl::sycl::range" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class.cl::sycl::range" addrspace(4)* %this1 to %"class.cl::sycl::detail::array" addrspace(4)*
  %1 = load i64, i64 addrspace(4)* %dim0.addr.ascast, align 8
  call spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(8) %0, i64 %1) #8
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(8) %this, i32 %dimension) #2 comdat align 2 {
entry:
  %this.addr.i = alloca %"class.cl::sycl::detail::array" addrspace(4)*, align 8
  %dimension.addr.i = alloca i32, align 4
  %retval = alloca i64 addrspace(4)*, align 8
  %this.addr = alloca %"class.cl::sycl::detail::array" addrspace(4)*, align 8
  %dimension.addr = alloca i32, align 4
  %retval.ascast = addrspacecast i64 addrspace(4)** %retval to i64 addrspace(4)* addrspace(4)*
  %this.addr.ascast = addrspacecast %"class.cl::sycl::detail::array" addrspace(4)** %this.addr to %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast = addrspacecast i32* %dimension.addr to i32 addrspace(4)*
  store %"class.cl::sycl::detail::array" addrspace(4)* %this, %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  store i32 %dimension, i32 addrspace(4)* %dimension.addr.ascast, align 4
  %this1 = load %"class.cl::sycl::detail::array" addrspace(4)*, %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = load i32, i32 addrspace(4)* %dimension.addr.ascast, align 4
  %this.addr.ascast.i = addrspacecast %"class.cl::sycl::detail::array" addrspace(4)** %this.addr.i to %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)*
  %dimension.addr.ascast.i = addrspacecast i32* %dimension.addr.i to i32 addrspace(4)*
  store %"class.cl::sycl::detail::array" addrspace(4)* %this1, %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  store i32 %0, i32 addrspace(4)* %dimension.addr.ascast.i, align 4
  %this1.i = load %"class.cl::sycl::detail::array" addrspace(4)*, %"class.cl::sycl::detail::array" addrspace(4)* addrspace(4)* %this.addr.ascast.i, align 8
  %common_array = getelementptr inbounds %"class.cl::sycl::detail::array", %"class.cl::sycl::detail::array" addrspace(4)* %this1, i32 0, i32 0
  %1 = load i32, i32 addrspace(4)* %dimension.addr.ascast, align 4
  %idxprom = sext i32 %1 to i64
  %arrayidx = getelementptr inbounds [1 x i64], [1 x i64] addrspace(4)* %common_array, i64 0, i64 %idxprom
  ret i64 addrspace(4)* %arrayidx
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) %"class.cl::sycl::id" addrspace(4)* @_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE9getOffsetEv(%"class.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(32) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class.cl::sycl::id" addrspace(4)*, align 8
  %this.addr = alloca %"class.cl::sycl::accessor" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class.cl::sycl::id" addrspace(4)** %retval to %"class.cl::sycl::id" addrspace(4)* addrspace(4)*
  %this.addr.ascast = addrspacecast %"class.cl::sycl::accessor" addrspace(4)** %this.addr to %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::accessor" addrspace(4)* %this, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::accessor" addrspace(4)*, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class.cl::sycl::accessor", %"class.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %Offset = getelementptr inbounds %"class.cl::sycl::detail::AccessorImplDevice", %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 0
  ret %"class.cl::sycl::id" addrspace(4)* %Offset
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) %"class.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getAccessRangeEv(%"class.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(32) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class.cl::sycl::range" addrspace(4)*, align 8
  %this.addr = alloca %"class.cl::sycl::accessor" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class.cl::sycl::range" addrspace(4)** %retval to %"class.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr.ascast = addrspacecast %"class.cl::sycl::accessor" addrspace(4)** %this.addr to %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::accessor" addrspace(4)* %this, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::accessor" addrspace(4)*, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class.cl::sycl::accessor", %"class.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %AccessRange = getelementptr inbounds %"class.cl::sycl::detail::AccessorImplDevice", %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 1
  ret %"class.cl::sycl::range" addrspace(4)* %AccessRange
}

; Function Attrs: convergent mustprogress noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func align 8 dereferenceable(8) %"class.cl::sycl::range" addrspace(4)* @_ZN2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getMemoryRangeEv(%"class.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(32) %this) #6 comdat align 2 {
entry:
  %retval = alloca %"class.cl::sycl::range" addrspace(4)*, align 8
  %this.addr = alloca %"class.cl::sycl::accessor" addrspace(4)*, align 8
  %retval.ascast = addrspacecast %"class.cl::sycl::range" addrspace(4)** %retval to %"class.cl::sycl::range" addrspace(4)* addrspace(4)*
  %this.addr.ascast = addrspacecast %"class.cl::sycl::accessor" addrspace(4)** %this.addr to %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  store %"class.cl::sycl::accessor" addrspace(4)* %this, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::accessor" addrspace(4)*, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %impl = getelementptr inbounds %"class.cl::sycl::accessor", %"class.cl::sycl::accessor" addrspace(4)* %this1, i32 0, i32 0
  %MemRange = getelementptr inbounds %"class.cl::sycl::detail::AccessorImplDevice", %"class.cl::sycl::detail::AccessorImplDevice" addrspace(4)* %impl, i32 0, i32 2
  ret %"class.cl::sycl::range" addrspace(4)* %MemRange
}

; Function Attrs: convergent noinline norecurse nounwind optnone
define linkonce_odr dso_local spir_func void @_ZN2cl4sycl2idILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::id" addrspace(4)* align 8 dereferenceable_or_null(8) %this, i64 %dim0) unnamed_addr #5 comdat align 2 {
entry:
  %this.addr = alloca %"class.cl::sycl::id" addrspace(4)*, align 8
  %dim0.addr = alloca i64, align 8
  %this.addr.ascast = addrspacecast %"class.cl::sycl::id" addrspace(4)** %this.addr to %"class.cl::sycl::id" addrspace(4)* addrspace(4)*
  %dim0.addr.ascast = addrspacecast i64* %dim0.addr to i64 addrspace(4)*
  store %"class.cl::sycl::id" addrspace(4)* %this, %"class.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  store i64 %dim0, i64 addrspace(4)* %dim0.addr.ascast, align 8
  %this1 = load %"class.cl::sycl::id" addrspace(4)*, %"class.cl::sycl::id" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class.cl::sycl::id" addrspace(4)* %this1 to %"class.cl::sycl::detail::array" addrspace(4)*
  %1 = load i64, i64 addrspace(4)* %dim0.addr.ascast, align 8
  call spir_func void @_ZN2cl4sycl6detail5arrayILi1EEC2ILi1EEENSt9enable_ifIXeqT_Li1EEmE4typeE(%"class.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(8) %0, i64 %1) #8
  ret void
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func i64 @_ZNK2cl4sycl8accessorINS0_6detail14AssertHappenedELi1ELNS0_6access4modeE1025ELNS4_6targetE2014ELNS4_11placeholderE0ENS0_3ext6oneapi22accessor_property_listIJEEEE14getLinearIndexILi1EEEmNS0_2idIXT_EEE(%"class.cl::sycl::accessor" addrspace(4)* align 8 dereferenceable_or_null(32) %this, %"class.cl::sycl::id"* byval(%"class.cl::sycl::id") align 8 %Id) #2 comdat align 2 {
entry:
  %retval = alloca i64, align 8
  %this.addr = alloca %"class.cl::sycl::accessor" addrspace(4)*, align 8
  %Result = alloca i64, align 8
  %retval.ascast = addrspacecast i64* %retval to i64 addrspace(4)*
  %this.addr.ascast = addrspacecast %"class.cl::sycl::accessor" addrspace(4)** %this.addr to %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)*
  %Result.ascast = addrspacecast i64* %Result to i64 addrspace(4)*
  store %"class.cl::sycl::accessor" addrspace(4)* %this, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %Id.ascast = addrspacecast %"class.cl::sycl::id"* %Id to %"class.cl::sycl::id" addrspace(4)*
  %this1 = load %"class.cl::sycl::accessor" addrspace(4)*, %"class.cl::sycl::accessor" addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %0 = bitcast %"class.cl::sycl::id" addrspace(4)* %Id.ascast to %"class.cl::sycl::detail::array" addrspace(4)*
  %call = call spir_func align 8 dereferenceable(8) i64 addrspace(4)* @_ZN2cl4sycl6detail5arrayILi1EEixEi(%"class.cl::sycl::detail::array" addrspace(4)* align 8 dereferenceable_or_null(8) %0, i32 0) #8
  %1 = load i64, i64 addrspace(4)* %call, align 8
  ret i64 %1
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define weak_odr dso_local spir_kernel void @_ZTSZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_EUlvE_() #0 comdat !kernel_arg_buffer_location !9 {
entry:
  %0 = alloca %class.anon.0, align 1
  %1 = addrspacecast %class.anon.0* %0 to %class.anon.0 addrspace(4)*
  call spir_func void @_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlvE_clEv(%class.anon.0 addrspace(4)* align 1 dereferenceable_or_null(1) %1) #8
  ret void
}

; CHECK-LABEL: define internal spir_func void @_ZZZ4main{{.*}}
; Function Attrs: convergent mustprogress noinline norecurse optnone
define internal spir_func void @_ZZZ4mainENKUlRN2cl4sycl7handlerEE_clES2_ENKUlvE_clEv(%class.anon.0 addrspace(4)* align 1 dereferenceable_or_null(1) %this) #2 align 2 {
entry:
  %this.addr = alloca %class.anon.0 addrspace(4)*, align 8
  %IntFormatString = alloca i8 addrspace(4)*, align 8
  %this.addr.ascast = addrspacecast %class.anon.0 addrspace(4)** %this.addr to %class.anon.0 addrspace(4)* addrspace(4)*
  %IntFormatString.ascast = addrspacecast i8 addrspace(4)** %IntFormatString to i8 addrspace(4)* addrspace(4)*
  store %class.anon.0 addrspace(4)* %this, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  %this1 = load %class.anon.0 addrspace(4)*, %class.anon.0 addrspace(4)* addrspace(4)* %this.addr.ascast, align 8
  ; In particular, make sure that no argument promotion has been done for float
  ; upon variadic redeclaration:
  ; CHECK: call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str._AS2, i32 0, i32 0), float 1.000000e+00)
  %call = call spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_(i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str to [15 x i8] addrspace(4)*), i64 0, i64 0), float 1.000000e+00) #8
  store i8 addrspace(4)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(4)* addrspacecast ([15 x i8] addrspace(1)* @.str.1 to [15 x i8] addrspace(4)*), i64 0, i64 0), i8 addrspace(4)* addrspace(4)* %IntFormatString.ascast, align 8
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %IntFormatString.ascast, align 8
  ; CHECK: call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 2)
  %call2 = call spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_(i8 addrspace(4)* %0, i32 2) #8
  %1 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %IntFormatString.ascast, align 8
  ; CHECK: call spir_func i32 (i8 addrspace(2)*, ...) @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)* getelementptr inbounds ([15 x i8], [15 x i8] addrspace(2)* @.str.1._AS2, i32 0, i32 0), i32 3)
  %call3 = call spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_(i8 addrspace(4)* %1, i32 3) #8
  ret void
}

; CHECK-LABEL: declare dso_local spir_func i32 @_Z18__spirv_ocl_printfPU3AS2Kcz(i8 addrspace(2)*, ...)

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJfEEEiPKT_DpT0_(i8 addrspace(4)* %__format, float %args) #2 comdat {
entry:
  %retval = alloca i32, align 4
  %__format.addr = alloca i8 addrspace(4)*, align 8
  %args.addr = alloca float, align 4
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %__format.addr.ascast = addrspacecast i8 addrspace(4)** %__format.addr to i8 addrspace(4)* addrspace(4)*
  %args.addr.ascast = addrspacecast float* %args.addr to float addrspace(4)*
  store i8 addrspace(4)* %__format, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  store float %args, float addrspace(4)* %args.addr.ascast, align 4
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  %1 = load float, float addrspace(4)* %args.addr.ascast, align 4
  %call = call spir_func i32 @_Z18__spirv_ocl_printfIJfEEiPKcDpT_(i8 addrspace(4)* %0, float %1) #8
  ret i32 %call
}

; Function Attrs: convergent mustprogress noinline norecurse optnone
define linkonce_odr dso_local spir_func i32 @_ZN2cl4sycl3ext6oneapi12experimental6printfIcJiEEEiPKT_DpT0_(i8 addrspace(4)* %__format, i32 %args) #2 comdat {
entry:
  %retval = alloca i32, align 4
  %__format.addr = alloca i8 addrspace(4)*, align 8
  %args.addr = alloca i32, align 4
  %retval.ascast = addrspacecast i32* %retval to i32 addrspace(4)*
  %__format.addr.ascast = addrspacecast i8 addrspace(4)** %__format.addr to i8 addrspace(4)* addrspace(4)*
  %args.addr.ascast = addrspacecast i32* %args.addr to i32 addrspace(4)*
  store i8 addrspace(4)* %__format, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  store i32 %args, i32 addrspace(4)* %args.addr.ascast, align 4
  %0 = load i8 addrspace(4)*, i8 addrspace(4)* addrspace(4)* %__format.addr.ascast, align 8
  %1 = load i32, i32 addrspace(4)* %args.addr.ascast, align 4
  %call = call spir_func i32 @_Z18__spirv_ocl_printfIJiEEiPKcDpT_(i8 addrspace(4)* %0, i32 %1) #8
  ret i32 %call
}

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJfEEiPKcDpT_(i8 addrspace(4)*, float) #7

; Function Attrs: convergent
declare dso_local spir_func i32 @_Z18__spirv_ocl_printfIJiEEiPKcDpT_(i8 addrspace(4)*, i32) #7

attributes #0 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "sycl-module-id"="experimental-printf.cpp" "uniform-work-group-size"="true" }
attributes #1 = { convergent noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { convergent mustprogress noinline norecurse optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #3 = { argmemonly nofree nounwind willreturn }
attributes #4 = { argmemonly nofree nounwind willreturn writeonly }
attributes #5 = { convergent noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #6 = { convergent mustprogress noinline norecurse nounwind optnone "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #7 = { convergent "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #8 = { convergent }

!llvm.module.flags = !{!0, !1}
!opencl.spir.version = !{!2}
!spirv.Source = !{!3}
!llvm.ident = !{!4}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"frame-pointer", i32 2}
!2 = !{i32 1, i32 2}
!3 = !{i32 4, i32 100000}
!4 = !{!"clang version 14.0.0"}
!5 = !{i32 -1, i32 -1, i32 -1, i32 -1}
!6 = distinct !{!6, !7, !8}
!7 = !{!"llvm.loop.mustprogress"}
!8 = !{!"llvm.loop.unroll.enable"}
!9 = !{}
