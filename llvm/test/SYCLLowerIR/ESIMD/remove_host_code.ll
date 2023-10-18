; RUN: opt -passes=esimd-remove-host-code -S < %s | FileCheck %s

; This test checks that ESIMDRemoveHostCode removes all code from ESIMD
; functions and leaves others untouched.

target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-unknown-linux-gnu"

; Function Attrs: alwaysinline mustprogress uwtable
define linkonce_odr dso_local void @foo() {
; CHECK: foo
; CHECK-NEXT: %1 = alloca double, align 8
; CHECK-NEXT: ret void
%1 = alloca double, align 8
ret void
}

; Function Attrs: alwaysinline mustprogress uwtable
define linkonce_odr dso_local void @_ZN4sycl3_V13ext5intel12experimental5esimd15lsc_block_storeIdLi64ELNS4_13lsc_data_sizeE0ELNS4_10cache_hintE0ELS7_0ENS2_5esimd6detail26dqword_element_aligned_tagEEENSt9enable_ifIXsr4sycl3ext5intel5esimdE19is_simd_flag_type_vIT4_EEvE4typeEPT_NS8_4simdISF_XT0_EEENS9_14simd_mask_implItLi1EEESC_() {
; CHECK: lsc_block_store
; CHECK-NEXT: ret void
%1 = alloca double, align 8
ret void
}

; Function Attrs: alwaysinline mustprogress uwtable
define linkonce_odr dso_local ptr @_ZN4sycl3_V13ext5intel12experimental5esimd15lsc_block_fobarIdLi64ELNS4_13lsc_data_sizeE0ELNS4_10cache_hintE0ELS7_0ENS2_5esimd6detail26dqword_element_aligned_tagEEENSt9enable_ifIXsr4sycl3ext5intel5esimdE19is_simd_flag_type_vIT4_EEvE4typeEPT_NS8_4simdISF_XT0_EEENS9_14simd_mask_implItLi1EEESC_() {
; CHECK: lsc_block_fobar
; CHECK-NEXT: ret ptr null
%1 = alloca double, align 8
ret ptr %1
}