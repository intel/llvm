; This test checks C++ ESIMD intrinsics lowering to "@llvm.genx.*" form
; consumable by the CM back-end.
;
; RUN: opt < %s -LowerESIMD -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown-sycldevice"

%opencl.image2d_ro_t = type opaque
%opencl.image2d_wo_t = type opaque

%"cm::gen::simd<int, 16>" = type { <16 x i32> }

@vg = dso_local global %"cm::gen::simd<int, 16>" zeroinitializer, align 64 #0
@vc = dso_local addrspace(1) global <32 x i32> zeroinitializer

; LowerESIMD pass should process every function, 
; !sycl_explicit_simd metadata is not necessary.
define dso_local spir_func <32 x i32>  @FUNC_1() {
  %a_1 = alloca <32 x i64>
  %1 = load <32 x i64>, <32 x i64>* %a_1
  %a_2 = alloca  <32 x i16>
  %2 = load  <32 x i16>,  <32 x i16>* %a_2
  %ret_val = call spir_func  <32 x i32>  @_Z20__esimd_flat_atomic0ILN2cm3gen14CmAtomicOpTypeE2EjLi32ELNS1_9CacheHintE0ELS3_0EENS1_13__vector_typeIT0_XT1_EE4typeENS4_IyXT1_EE4typeENS4_ItXT1_EE4typeE(<32 x i64> %1,  <32 x i16> %2)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.inc.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)
  ret <32 x i32>  %ret_val
}

define dso_local spir_func <32 x i32>  @FUNC_2() {
  %a_1 = alloca <32 x i64>
  %1 = load <32 x i64>, <32 x i64>* %a_1
  %a_2 = alloca  <32 x i32>
  %2 = load  <32 x i32>,  <32 x i32>* %a_2
  %a_3 = alloca  <32 x i16>
  %3 = load  <32 x i16>,  <32 x i16>* %a_3
  %ret_val = call spir_func  <32 x i32>  @_Z20__esimd_flat_atomic1ILN2cm3gen14CmAtomicOpTypeE0EjLi32ELNS1_9CacheHintE0ELS3_0EENS1_13__vector_typeIT0_XT1_EE4typeENS4_IyXT1_EE4typeES7_NS4_ItXT1_EE4typeE(<32 x i64> %1,  <32 x i32> %2,  <32 x i16> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.add.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)
  ret <32 x i32>  %ret_val
}

define dso_local spir_func <32 x i32>  @FUNC_3() {
  %a_1 = alloca <32 x i64>
  %1 = load <32 x i64>, <32 x i64>* %a_1
  %a_2 = alloca  <32 x i32>
  %2 = load  <32 x i32>,  <32 x i32>* %a_2
  %a_3 = alloca  <32 x i32>
  %3 = load  <32 x i32>,  <32 x i32>* %a_3
  %a_4 = alloca  <32 x i16>
  %4 = load  <32 x i16>,  <32 x i16>* %a_4
  %ret_val = call spir_func  <32 x i32>  @_Z20__esimd_flat_atomic2ILN2cm3gen14CmAtomicOpTypeE7EjLi32ELNS1_9CacheHintE0ELS3_0EENS1_13__vector_typeIT0_XT1_EE4typeENS4_IyXT1_EE4typeES7_S7_NS4_ItXT1_EE4typeE(<32 x i64> %1,  <32 x i32> %2,  <32 x i32> %3,  <32 x i16> %4)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.atomic.cmpxchg.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)
  ret <32 x i32>  %ret_val
}

define dso_local spir_func <32 x i32>  @FUNC_4() {
  %ret_val = call spir_func  <32 x i32>  @_Z33__esimd_flat_block_read_unalignedIjLi32ELN2cm3gen9CacheHintE0ELS2_0EENS1_13__vector_typeIT_XT0_EE4typeEy(i64 0)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.block.ld.unaligned.v32i32.i64(i64 0)
  ret <32 x i32>  %ret_val
}

define dso_local spir_func void  @FUNC_5() {
  %a_1 = alloca  <32 x i32>
  %1 = load  <32 x i32>,  <32 x i32>* %a_1
  call spir_func  void  @_Z24__esimd_flat_block_writeIjLi32ELN2cm3gen9CacheHintE0ELS2_0EEvyNS1_13__vector_typeIT_XT0_EE4typeE(i64 0,  <32 x i32> %1)
; CHECK: call void @llvm.genx.svm.block.st.i64.v32i32(i64 0, <32 x i32> %{{[0-9a-zA-Z_.]+}})
  ret void
}

define dso_local spir_func <32 x i32>  @FUNC_6() {
  %a_1 = alloca <32 x i64>
  %1 = load <32 x i64>, <32 x i64>* %a_1
  %a_2 = alloca  <32 x i16>
  %2 = load  <32 x i16>,  <32 x i16>* %a_2
  %ret_val = call spir_func  <32 x i32>  @_Z17__esimd_flat_readIjLi32ELi0ELN2cm3gen9CacheHintE0ELS2_0EENS1_13__vector_typeIT_XmlT0_clL_ZNS1_20ElemsPerAddrDecodingEjET1_EEE4typeENS3_IyXT0_EE4typeEiNS3_ItXT0_EE4typeE(<32 x i64> %1, i32 0, <32 x i16> %2)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.svm.gather.v32i32.v32i1.v32i64(<32 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> undef)
  ret <32 x i32>  %ret_val
}

define dso_local spir_func void  @FUNC_7() {
  %a_1 = alloca <32 x i64>
  %1 = load <32 x i64>, <32 x i64>* %a_1
  %a_2 = alloca  <32 x i32>
  %2 = load  <32 x i32>,  <32 x i32>* %a_2
  %a_3 = alloca  <32 x i16>
  %3 = load  <32 x i16>,  <32 x i16>* %a_3
  call spir_func  void  @_Z18__esimd_flat_writeIjLi32ELi0ELN2cm3gen9CacheHintE0ELS2_0EEvNS1_13__vector_typeIyXT0_EE4typeENS3_IT_XmlT0_clL_ZNS1_20ElemsPerAddrDecodingEjET1_EEE4typeEiNS3_ItXT0_EE4typeE(<32 x i64> %1,  <32 x i32> %2, i32 0, <32 x i16> %3)
; CHECK: call void @llvm.genx.svm.scatter.v32i1.v32i64.v32i32(<32 x i1> %{{[0-9a-zA-Z_.]+}}, i32 0, <32 x i64> %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}})
  ret void
}

define dso_local spir_func <16 x i16>  @FUNC_8() {
  %a_1 = alloca <16 x i16>
  %1 = load <16 x i16>, <16 x i16>* %a_1
  %a_2 = alloca  <16 x i16>
  %2 = load  <16 x i16>,  <16 x i16>* %a_2
  %ret_val = call spir_func  <16 x i16>  @_Z12__esimd_sminIsLi16EEN2cm3gen13__vector_typeIT_XT0_EE4typeES5_S5_(<16 x i16> %1,  <16 x i16> %2)
; CHECK:  %{{[0-9a-zA-Z_.]+}} = call <16 x i16> @llvm.genx.smin.v16i16.v16i16(<16 x i16> %{{[0-9a-zA-Z_.]+}}, <16 x i16> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i16>  %ret_val
}

define dso_local spir_func <1 x float>  @FUNC_9() {
  %a_1 = alloca <1 x float>
  %1 = load <1 x float>, <1 x float>* %a_1
  %a_2 = alloca  <1 x float>
  %2 = load  <1 x float>,  <1 x float>* %a_2
  %ret_val = call spir_func  <1 x float>  @_Z16__esimd_div_ieeeILi1EEN2cm3gen13__vector_typeIfXT_EE4typeES4_S4_(<1 x float> %1,  <1 x float> %2)
; CHECK:  %{{[0-9a-zA-Z_.]+}} = call <1 x float> @llvm.genx.ieee.div.v1f32(<1 x float>  %{{[0-9a-zA-Z_.]+}}, <1 x float>  %{{[0-9a-zA-Z_.]+}})
  ret <1 x float>  %ret_val
}

define dso_local spir_func <8 x float>  @FUNC_10() {
  %a_1 = alloca <16 x float>
  %1 = load <16 x float>, <16 x float>* %a_1
  %ret_val = call spir_func  <8 x float>  @_Z16__esimd_rdregionIfLi16ELi8ELi0ELi8ELi1ELi0EEN2cm3gen13__vector_typeIT_XT1_EE4typeENS2_IS3_XT0_EE4typeEt(<16 x float> %1, i16 zeroext 0)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x float> @llvm.genx.rdregionf.v8f32.v16f32.i16(<16 x float> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 1, i16 0, i32 0)
  ret <8 x float>  %ret_val
}

define dso_local spir_func <16 x float>  @FUNC_11() {
  %a_1 = alloca <16 x float>
  %1 = load <16 x float>, <16 x float>* %a_1
  %a_2 = alloca  <8 x float>
  %2 = load  <8 x float>,  <8 x float>* %a_2
  %ret_val = call spir_func  <16 x float>  @_Z16__esimd_wrregionIfLi16ELi8ELi0ELi8ELi1ELi0EEN2cm3gen13__vector_typeIT_XT0_EE4typeES5_NS2_IS3_XT1_EE4typeEtNS2_ItXT1_EE4typeE(<16 x float> %1,  <8 x float> %2, i16 zeroext 0, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.wrregionf.v16f32.v8f32.i16.v8i1(<16 x float> %{{[0-9a-zA-Z_.]+}}, <8 x float> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 1, i16 0, i32 0, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret <16 x float>  %ret_val
}

define dso_local spir_func <32 x i32>  @FUNC_21(%opencl.image2d_ro_t addrspace(1)* %0, i32 %1, i32 %2) {
  %ret_val = call spir_func  <32 x i32>  @_Z24__esimd_media_block_loadIiLi4ELi8E14ocl_image2d_roEN2cm3gen13__vector_typeIT_XmlT0_T1_EE4typeEjT2_jjjj(i32 0, %opencl.image2d_ro_t addrspace(1)* %0, i32 0, i32 32, i32 %1, i32 %2)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <32 x i32> @llvm.genx.media.ld.v32i32(i32 0, i32 %{{[0-9a-zA-Z_.]+}}, i32 0, i32 32, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}})
  ret <32 x i32>  %ret_val
}

define dso_local spir_func void  @FUNC_22(%opencl.image2d_wo_t addrspace(1)* %0, i32 %1, i32 %2) {
  %a_3 = alloca <32 x i32>
  %4 = load <32 x i32>, <32 x i32>* %a_3
  call spir_func  void  @_Z25__esimd_media_block_storeIiLi4ELi8E14ocl_image2d_woEvjT2_jjjjN2cm3gen13__vector_typeIT_XmlT0_T1_EE4typeE(i32 0, %opencl.image2d_wo_t addrspace(1)* %0, i32 0, i32 32, i32 %1, i32 %2, <32 x i32> %4)
; CHECK: call void @llvm.genx.media.st.v32i32(i32 0, i32 %{{[0-9a-zA-Z_.]+}}, i32 0, i32 32, i32 %{{[0-9a-zA-Z_.]+}}, i32 %{{[0-9a-zA-Z_.]+}}, <32 x i32> %{{[0-9a-zA-Z_.]+}})
  ret void
}

define dso_local spir_func <16 x i32>  @FUNC_23() {
  %ret_val = call spir_func <16 x i32> @_Z13__esimd_vloadIiLi16EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<16 x i32> addrspace(4)* addrspacecast (<16 x i32>* getelementptr inbounds (%"cm::gen::simd<int, 16>", %"cm::gen::simd<int, 16>"* @vg, i32 0, i32 0) to <16 x i32> addrspace(4)*))
; CHECK: %ret_val1 = load <16 x i32>, <16 x i32> addrspace(4)* addrspacecast (<16 x i32>* getelementptr inbounds (%"cm::gen::simd<int, 16>", %"cm::gen::simd<int, 16>"* @vg, i32 0, i32 0) to <16 x i32> addrspace(4)*), align 64
; TODO: testcase to generate this:
; CxHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.vload.v16i32.p4v16i32(<16 x i32> addrspace(4)* {{.*}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func void  @FUNC_28(<32 x i32> %0) {
  call spir_func void @_Z14__esimd_vstoreIiLi32EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<32 x i32> addrspace(4)* addrspacecast (<32 x i32> addrspace(1)* @vc to <32 x i32> addrspace(4)*), <32 x i32> %0)
; CHECK:  store <32 x i32> %0, <32 x i32> addrspace(4)* addrspacecast (<32 x i32> addrspace(1)* @vc to <32 x i32> addrspace(4)*), align 128

  ret void
}

define dso_local spir_func void  @FUNC_29() {
  %a_1 = alloca <32 x i32>
  %1 = addrspacecast <32 x i32>* %a_1 to <32 x i32>  addrspace(4)*
  %a_2 = alloca  <32 x i32>
  %2 = load  <32 x i32>,  <32 x i32>* %a_2
  call spir_func  void  @_Z14__esimd_vstoreIiLi32EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<32 x i32> addrspace(4)* %1,  <32 x i32> %2)
; CHECK: store <32 x i32>  %{{[0-9a-zA-Z_.]+}}, <32 x i32> addrspace(4)* {{.*}}
  ret void
}

define dso_local spir_kernel void  @FUNC_30() {
; CHECK: define dso_local spir_kernel void  @FUNC_30()
  call spir_func void @_ZN2cl4sycl3ext5intel12experimental5esimd8slm_initEj(i32 1023)
  ret void
; CHECK-NEXT: ret void
}

define dso_local spir_func <16 x i32>  @FUNC_32() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z14__esimd_uudp4aIjjjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.uudp4a.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_33() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z14__esimd_usdp4aIjiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.usdp4a.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_34() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z14__esimd_sudp4aIijjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.sudp4a.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_35() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z14__esimd_ssdp4aIiiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.ssdp4a.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_36() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z18__esimd_uudp4a_satIjjjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.uudp4a.sat.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_37() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z18__esimd_usdp4a_satIjiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.usdp4a.sat.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_38() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z18__esimd_sudp4a_satIijjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.sudp4a.sat.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_39() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <16 x i32>
  %2 = load  <16 x i32>,  <16 x i32>* %a_2
  %a_3 = alloca  <16 x i32>
  %3 = load  <16 x i32>,  <16 x i32>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z18__esimd_ssdp4a_satIiiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %1, <16 x i32> %2, <16 x i32> %3)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x i32> @llvm.genx.ssdp4a.sat.v16i32.v16i32.v16i32.v16i32(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x i32> %{{[0-9a-zA-Z_.]+}})
  ret <16 x i32>  %ret_val
}

define dso_local spir_func <8 x i32>  @FUNC_40() {
  %ret_val = call spir_func  <8 x i32>  @_Z22__esimd_slm_block_readIiLi8EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT0_EE4typeEj(i32 0)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x i32> @llvm.genx.oword.ld.v8i32(i32 0, i32 254, i32 0)
  ret <8 x i32>  %ret_val
}

define dso_local spir_func void  @FUNC_41() {
  call spir_func void @_Z16__esimd_sbarrierN2cl4sycl5INTEL3gpu17EsimdSbarrierTypeE(i8 zeroext 1)
; CHECK: call void @llvm.genx.sbarrier(i8 1)
  ret void
}

define dso_local spir_func void  @FUNC_42() {
  call spir_func void @_Z16__esimd_sbarrierN2cl4sycl5INTEL3gpu17EsimdSbarrierTypeE(i8 zeroext 0)
; CHECK: call void @llvm.genx.sbarrier(i8 0)
  ret void
}

define dso_local spir_func <8 x i32>  @FUNC_43() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca <8 x i16>
  %2 = load <8 x i16>, <8 x i16>* %a_2
  %ret_val = call spir_func <8 x i32> @_Z18__esimd_rdindirectIiLi16ELi8ELi0EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT1_EE4typeENS4_IS5_XT0_EE4typeENS4_ItXT1_EE4typeE(<16 x i32> %1, <8 x i16> %2)
; CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x i32> @llvm.genx.rdregioni.v8i32.v16i32.v8i16(<16 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 0, <8 x i16> %{{[0-9a-zA-Z_.]+}}, i32 0)
  ret <8 x i32>  %ret_val
}

define dso_local spir_func <16 x i32>  @FUNC_44() {
  %a_1 = alloca <16 x i32>
  %1 = load <16 x i32>, <16 x i32>* %a_1
  %a_2 = alloca  <8 x i32>
  %2 = load  <8 x i32>,  <8 x i32>* %a_2
  %a_3 = alloca  <8 x i16>
  %3 = load  <8 x i16>,  <8 x i16>* %a_3
  %ret_val = call spir_func <16 x i32> @_Z18__esimd_wrindirectIiLi16ELi8ELi0EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT0_EE4typeES7_NS4_IS5_XT1_EE4typeENS4_ItXT1_EE4typeESB_(<16 x i32> %1, <8 x i32> %2, <8 x i16> %3, <8 x i16> <i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1, i16 1>)
; CHECK: %{{[0-9a-zA-Z_.]+}} =  call <16 x i32> @llvm.genx.wrregioni.v16i32.v8i32.v8i16.v8i1(<16 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x i32> %{{[0-9a-zA-Z_.]+}}, i32 0, i32 8, i32 0, <8 x i16> %{{[0-9a-zA-Z_.]+}}, i32 0, <8 x i1> <i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true, i1 true>)
  ret <16 x i32>  %ret_val
}

; TODO LowerESIMD.cpp temporarily removes @llvm.assume, this test checks this.
; Remove once @llvm.assume is allowed in the ESIMD BE.
define dso_local spir_func void  @FUNC_45() {
; CHECK-LABEL: FUNC_45
  call void @llvm.assume(i1 1)
; CHECK-NOT: @llvm.assume
  ret void
}
; CHECK-LABEL: }

declare void @llvm.assume(i1 noundef)

define dso_local i32 @FUNC_46() {
; CHECK-LABEL: FUNC_46
; CHECK: %{{[0-9a-zA-Z_.]+}} = call i32 @llvm.genx.lane.id()
  %call = call i32 @_Z15__esimd_lane_idv()
  ret i32 %call
}

define dso_local spir_func <16 x float>  @FUNC_47() {
  %a_1 = alloca <16 x float>
  %1 = load <16 x float>, <16 x float>* %a_1
  %ret_val = call spir_func  <16 x float> @_Z12__esimd_rnddILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float> %1)
; CHECK:  %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.rndd.v16f32(<16 x float> %{{[0-9a-zA-Z_.]+}})
  ret <16 x float>  %ret_val
}

define dso_local spir_func <16 x float>  @FUNC_48() {
  %a_1 = alloca <16 x float>
  %1 = load <16 x float>, <16 x float>* %a_1
  %ret_val = call spir_func  <16 x float> @_Z12__esimd_rnduILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float> %1)
; CHECK:  %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.rndu.v16f32(<16 x float> %{{[0-9a-zA-Z_.]+}})
  ret <16 x float>  %ret_val
}

define dso_local spir_func <16 x float>  @FUNC_49() {
  %a_1 = alloca <16 x float>
  %1 = load <16 x float>, <16 x float>* %a_1
  %ret_val = call spir_func  <16 x float> @_Z12__esimd_rndzILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float> %1)
; CHECK:  %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.rndz.v16f32(<16 x float> %{{[0-9a-zA-Z_.]+}})
  ret <16 x float>  %ret_val
}

define dso_local spir_func <16 x float>  @FUNC_50() {
  %a_1 = alloca <16 x float>
  %1 = load <16 x float>, <16 x float>* %a_1
  %ret_val = call spir_func  <16 x float> @_Z12__esimd_rndeILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float> %1)
; CHECK:  %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.rnde.v16f32(<16 x float> %{{[0-9a-zA-Z_.]+}})
  ret <16 x float>  %ret_val
}

declare dso_local i32 @_Z15__esimd_lane_idv()

declare dso_local spir_func <32 x i32> @_Z20__esimd_flat_atomic0ILN2cm3gen14CmAtomicOpTypeE2EjLi32ELNS1_9CacheHintE0ELS3_0EENS1_13__vector_typeIT0_XT1_EE4typeENS4_IyXT1_EE4typeENS4_ItXT1_EE4typeE(<32 x i64> %0, <32 x i16> %1)
declare dso_local spir_func <32 x i32> @_Z20__esimd_flat_atomic1ILN2cm3gen14CmAtomicOpTypeE0EjLi32ELNS1_9CacheHintE0ELS3_0EENS1_13__vector_typeIT0_XT1_EE4typeENS4_IyXT1_EE4typeES7_NS4_ItXT1_EE4typeE(<32 x i64> %0, <32 x i32> %1, <32 x i16> %2)
declare dso_local spir_func <32 x i32> @_Z20__esimd_flat_atomic2ILN2cm3gen14CmAtomicOpTypeE7EjLi32ELNS1_9CacheHintE0ELS3_0EENS1_13__vector_typeIT0_XT1_EE4typeENS4_IyXT1_EE4typeES7_S7_NS4_ItXT1_EE4typeE(<32 x i64> %0, <32 x i32> %1, <32 x i32> %2, <32 x i16> %3)
declare dso_local spir_func <32 x i32> @_Z33__esimd_flat_block_read_unalignedIjLi32ELN2cm3gen9CacheHintE0ELS2_0EENS1_13__vector_typeIT_XT0_EE4typeEy(i64 %0)
declare dso_local spir_func void @_Z24__esimd_flat_block_writeIjLi32ELN2cm3gen9CacheHintE0ELS2_0EEvyNS1_13__vector_typeIT_XT0_EE4typeE(i64 %0, <32 x i32> %1)
declare dso_local spir_func <32 x i32> @_Z17__esimd_flat_readIjLi32ELi0ELN2cm3gen9CacheHintE0ELS2_0EENS1_13__vector_typeIT_XmlT0_clL_ZNS1_20ElemsPerAddrDecodingEjET1_EEE4typeENS3_IyXT0_EE4typeEiNS3_ItXT0_EE4typeE(<32 x i64> %0, i32 %1, <32 x i16> %2)
declare dso_local spir_func void @_Z18__esimd_flat_writeIjLi32ELi0ELN2cm3gen9CacheHintE0ELS2_0EEvNS1_13__vector_typeIyXT0_EE4typeENS3_IT_XmlT0_clL_ZNS1_20ElemsPerAddrDecodingEjET1_EEE4typeEiNS3_ItXT0_EE4typeE(<32 x i64> %0, <32 x i32> %1, i32 %2, <32 x i16> %3)
declare dso_local spir_func <16 x i16> @_Z12__esimd_sminIsLi16EEN2cm3gen13__vector_typeIT_XT0_EE4typeES5_S5_(<16 x i16> %0, <16 x i16> %1)
declare dso_local spir_func <1 x float> @_Z16__esimd_div_ieeeILi1EEN2cm3gen13__vector_typeIfXT_EE4typeES4_S4_(<1 x float> %0, <1 x float> %1)
declare dso_local spir_func <8 x float> @_Z16__esimd_rdregionIfLi16ELi8ELi0ELi8ELi1ELi0EEN2cm3gen13__vector_typeIT_XT1_EE4typeENS2_IS3_XT0_EE4typeEt(<16 x float> %0, i16 zeroext %1)
declare dso_local spir_func <16 x float> @_Z16__esimd_wrregionIfLi16ELi8ELi0ELi8ELi1ELi0EEN2cm3gen13__vector_typeIT_XT0_EE4typeES5_NS2_IS3_XT1_EE4typeEtNS2_ItXT1_EE4typeE(<16 x float> %0, <8 x float> %1, i16 zeroext %2, <8 x i16> %3)
declare dso_local spir_func <16 x i32> @_Z13__esimd_vloadIiLi16EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<16 x i32> addrspace(4)* %0)
declare dso_local spir_func void @_Z14__esimd_vstoreIiLi32EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<32 x i32> addrspace(4)* %0, <32 x i32> %1)
declare dso_local spir_func void @_Z14__esimd_vstoreIyLi32EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<32 x i64> addrspace(4)* %0, <32 x i64> %1)
declare dso_local spir_func <32 x i64> @_Z13__esimd_vloadIyLi32EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<32 x i64> addrspace(4)* %0)
declare dso_local spir_func <32 x i16> @_Z13__esimd_vloadItLi32EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<32 x i16> addrspace(4)* %0)
declare dso_local spir_func void @_Z14__esimd_vstoreIjLi32EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<32 x i32> addrspace(4)* %0, <32 x i32> %1)
declare dso_local spir_func <32 x i32> @_Z13__esimd_vloadIjLi32EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<32 x i32> addrspace(4)* %0)
declare dso_local spir_func <16 x i16> @_Z13__esimd_vloadIsLi16EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<16 x i16> addrspace(4)* %0)
declare dso_local spir_func void @_Z14__esimd_vstoreIsLi16EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<16 x i16> addrspace(4)* %0, <16 x i16> %1)
declare dso_local spir_func <1 x float> @_Z13__esimd_vloadIfLi1EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<1 x float> addrspace(4)* %0)
declare dso_local spir_func void @_Z14__esimd_vstoreIfLi1EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<1 x float> addrspace(4)* %0, <1 x float> %1)
declare dso_local spir_func <16 x float> @_Z13__esimd_vloadIfLi16EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<16 x float> addrspace(4)* %0)
declare dso_local spir_func void @_Z14__esimd_vstoreIfLi8EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<8 x float> addrspace(4)* %0, <8 x float> %1)
declare dso_local spir_func <8 x float> @_Z13__esimd_vloadIfLi8EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<8 x float> addrspace(4)* %0)
declare dso_local spir_func <32 x i32> @_Z24__esimd_media_block_loadIiLi4ELi8E14ocl_image2d_roEN2cm3gen13__vector_typeIT_XmlT0_T1_EE4typeEjT2_jjjj(i32 %0, %opencl.image2d_ro_t addrspace(1)* %1, i32 %2, i32 %3, i32 %4, i32 %5)
declare dso_local spir_func void @_Z25__esimd_media_block_storeIiLi4ELi8E14ocl_image2d_woEvjT2_jjjjN2cm3gen13__vector_typeIT_XmlT0_T1_EE4typeE(i32 %0, %opencl.image2d_wo_t addrspace(1)* %1, i32 %2, i32 %3, i32 %4, i32 %5, <32 x i32> %6)
declare dso_local spir_func <32 x i32> @_Z13__esimd_vloadIiLi32EEN2cm3gen13__vector_typeIT_XT0_EE4typeEPKS5_(<32 x i32> addrspace(4)* %0)
declare dso_local spir_func void @_Z14__esimd_vstoreIfLi16EEvPN2cm3gen13__vector_typeIT_XT0_EE4typeES5_(<16 x float> addrspace(4)* %0, <16 x float> %1)
declare dso_local spir_func void @_ZN2cl4sycl3ext5intel12experimental5esimd8slm_initEj(i32)
declare dso_local spir_func <16 x i32> @_Z14__esimd_uudp4aIjjjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <16 x i32> @_Z14__esimd_usdp4aIjiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <16 x i32> @_Z14__esimd_sudp4aIijjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <16 x i32> @_Z14__esimd_ssdp4aIiiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <16 x i32> @_Z18__esimd_uudp4a_satIjjjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <16 x i32> @_Z18__esimd_usdp4a_satIjiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <16 x i32> @_Z18__esimd_sudp4a_satIijjjLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <16 x i32> @_Z18__esimd_ssdp4a_satIiiiiLi16EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT3_EE4typeENS4_IT0_XT3_EE4typeENS4_IT1_XT3_EE4typeENS4_IT2_XT3_EE4typeE(<16 x i32> %0, <16 x i32> %1, <16 x i32> %2)
declare dso_local spir_func <8 x i32> @_Z22__esimd_slm_block_readIiLi8EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT0_EE4typeEj(i32 %0)
declare dso_local spir_func void @_Z16__esimd_sbarrierN2cl4sycl5INTEL3gpu17EsimdSbarrierTypeE(i8 %0)
declare dso_local spir_func <8 x i32> @_Z18__esimd_rdindirectIiLi16ELi8ELi0EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT1_EE4typeENS4_IS5_XT0_EE4typeENS4_ItXT1_EE4typeE(<16 x i32>, <8 x i16>)
declare dso_local spir_func <16 x i32> @_Z18__esimd_wrindirectIiLi16ELi8ELi0EEN2cl4sycl5INTEL3gpu11vector_typeIT_XT0_EE4typeES7_NS4_IS5_XT1_EE4typeENS4_ItXT1_EE4typeESB_(<16 x i32>, <8 x i32>, <8 x i16>, <8 x i16>)
declare dso_local spir_func <16 x float> @_Z12__esimd_rnddILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float>)
declare dso_local spir_func <16 x float> @_Z12__esimd_rnduILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float>)
declare dso_local spir_func <16 x float> @_Z12__esimd_rndzILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float>)
declare dso_local spir_func <16 x float> @_Z12__esimd_rndeILi16EEN2cl4sycl3ext5intel12experimental5esimd6detail11vector_typeIfXT_EE4typeES9_(<16 x float>)

attributes #0 = { "genx_byte_offset"="192" "genx_volatile" }

!genx.kernels = !{!0}

!0 = !{void ()* @"FUNC_30", !"FUNC_30", !1, i32 0, i32 0, !1, !2, i32 0, i32 0}
; CHECK: !0 = !{void ()* @FUNC_30, !"FUNC_30", !1, i32 1023, i32 0, !1, !2, i32 0, i32 0}
!1 = !{i32 0, i32 0}
!2 = !{}
