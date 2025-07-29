; This test verifies that we propagate the ESIMD attribute to a function that
; doesn't call any ESIMD-attribute functions but calls an ESIMD intrinsic

; RUN: opt -passes=lower-esimd-kernel-attrs -S < %s | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; CHECK: define dso_local spir_func void @FUNC() !sycl_explicit_simd
define dso_local spir_func void @FUNC() {
 %a_1 = alloca <16 x float>
  %1 = load <16 x float>, ptr %a_1
  %ret_val = call spir_func  <8 x float>  @_Z16__esimd_rdregionIfLi16ELi8ELi0ELi8ELi1ELi0EEN2cm3gen13__vector_typeIT_XT1_EE4typeENS2_IS3_XT0_EE4typeEt(<16 x float> %1, i16 zeroext 0)
  ret void
}

declare dso_local spir_func <8 x float> @_Z16__esimd_rdregionIfLi16ELi8ELi0ELi8ELi1ELi0EEN2cm3gen13__vector_typeIT_XT1_EE4typeENS2_IS3_XT0_EE4typeEt(<16 x float> %0, i16 zeroext %1)
