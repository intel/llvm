; RUN: opt < %s -passes=LowerESIMD -S | FileCheck %s

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64"
target triple = "spir64-unknown-unknown"

; Function Attrs: convergent norecurse mustprogress
define dso_local spir_kernel void @_ZTSZ6calleriE12kernel_esimd() !sycl_explicit_simd !3 {
entry:
  %0 = add <16 x i16> zeroinitializer, zeroinitializer
  %1 = add <8 x i32> zeroinitializer, zeroinitializer
  %2 = fadd <16 x float> zeroinitializer, zeroinitializer
  %3 = fadd <8 x float> zeroinitializer, zeroinitializer
  %4 = add <8 x i16> zeroinitializer, zeroinitializer
  %5 = add <8 x i64> zeroinitializer, zeroinitializer
  
  ; CHECK: %{{[0-9a-zA-Z_.]+}} = call <16 x float> @llvm.genx.raw.send2.v16f32.v16i1.v8i32(i8 0, i8 -124, <16 x i1> %{{[0-9a-zA-Z_.]+}}, i8 1, i8 2, i8 0, i32 10, i32 0, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <16 x float> %{{[0-9a-zA-Z_.]+}})
  %6 = call spir_func noundef <16 x float> @_Z27__esimd_raw_send2_constexprILh0ELh132ELh1ELh2ELh0EfLi16EjLi8ELi16EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT4_XT5_EE4typeENS6_ItXT8_EE4typeEjjNS6_IT6_XT7_EE4typeES9_(<16 x i16> noundef %0, i32 noundef 10, i32 noundef 0, <8 x i32> noundef %1, <16 x float> noundef %2) #1
  
  ; CHECK: call void @llvm.genx.raw.send2.noresult.v8i1.v8i32(i8 0, i8 -125, <8 x i1> %{{[0-9a-zA-Z_.]+}}, i8 2, i8 1, i32 10, i32 0, <8 x i32> %{{[0-9a-zA-Z_.]+}})
  call spir_func void @_Z36__esimd_raw_send2_noresult_constexprILh0ELh131ELh2ELh1EjLi8ELi8EEvN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeItXT5_EE4typeEjjNS6_IT3_XT4_EE4typeE(<8 x i16> noundef %4, i32 noundef 10, i32 noundef 0, <8 x i32> noundef %1)

  ; CHECK: %{{[0-9a-zA-Z_.]+}} = call <8 x i32> @llvm.genx.raw.sends2.v8i32.v8i1.v8i64.v8i32(i8 0, i8 -125, <8 x i1> %{{[0-9a-zA-Z_.]+}}, i8 2, i8 1, i8 1, i8 1, i32 0, i32 0, <8 x i64> %{{[0-9a-zA-Z_.]+}}, <8 x i32> %1, <8 x i32> %{{[0-9a-zA-Z_.]+}})
  %7 = call spir_func noundef <8 x i32> @_Z28__esimd_raw_sends2_constexprILh0ELh131ELh2ELh1ELh1ELh1EjLi8EmLi8EjLi8ELi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT5_XT6_EE4typeENS6_ItXT11_EE4typeEjjNS6_IT7_XT8_EE4typeENS6_IT9_XT10_EE4typeES9_(<8 x i16> noundef %4, i32 noundef 0, i32 noundef 0, <8 x i64> noundef %5, <8 x i32> noundef %1, <8 x i32> %1)

  ; CHECK: call void @llvm.genx.raw.sends2.noresult.v16i1.v8i32.v8f32(i8 0, i8 -125, <16 x i1> %{{[0-9a-zA-Z_.]+}}, i8 1, i8 1, i8 0, i32 10, i32 0, <8 x i32> %{{[0-9a-zA-Z_.]+}}, <8 x float> %{{[0-9a-zA-Z_.]+}})
  call spir_func void @_Z37__esimd_raw_sends2_noresult_constexprILh0ELh131ELh1ELh1ELh0EjLi8EfLi8ELi16EEvN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeItXT8_EE4typeEjjNS6_IT4_XT5_EE4typeENS6_IT6_XT7_EE4typeE(<16 x i16>  %0, i32  10, i32  0, <8 x i32>  %1, <8 x float> %3)
  
  ret void
}
!3 = !{}

declare dso_local spir_func noundef <16 x float> @_Z27__esimd_raw_send2_constexprILh0ELh132ELh1ELh2ELh0EfLi16EjLi8ELi16EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT4_XT5_EE4typeENS6_ItXT8_EE4typeEjjNS6_IT6_XT7_EE4typeES9_(<16 x i16> noundef, i32 noundef, i32 noundef, <8 x i32> noundef, <16 x float> noundef) local_unnamed_addr #1
declare dso_local spir_func void @_Z36__esimd_raw_send2_noresult_constexprILh0ELh131ELh2ELh1EjLi8ELi8EEvN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeItXT5_EE4typeEjjNS6_IT3_XT4_EE4typeE(<8 x i16> noundef, i32 noundef, i32 noundef, <8 x i32> noundef) local_unnamed_addr #1
declare dso_local spir_func noundef <8 x i32> @_Z28__esimd_raw_sends2_constexprILh0ELh131ELh2ELh1ELh1ELh1EjLi8EmLi8EjLi8ELi8EEN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeIT5_XT6_EE4typeENS6_ItXT11_EE4typeEjjNS6_IT7_XT8_EE4typeENS6_IT9_XT10_EE4typeES9_(<8 x i16> noundef, i32 noundef, i32 noundef, <8 x i64> noundef, <8 x i32> noundef, <8 x i32> noundef) local_unnamed_addr #1
declare dso_local spir_func void @_Z37__esimd_raw_sends2_noresult_constexprILh0ELh131ELh1ELh1ELh0EjLi8EfLi8ELi16EEvN4sycl3_V13ext5intel5esimd6detail15raw_vector_typeItXT8_EE4typeEjjNS6_IT4_XT5_EE4typeENS6_IT6_XT7_EE4typeE(<16 x i16> noundef, i32 noundef, i32 noundef, <8 x i32> noundef, <8 x float> noundef) local_unnamed_addr #1
