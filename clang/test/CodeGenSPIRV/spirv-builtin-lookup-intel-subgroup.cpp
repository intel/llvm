// RUN: %clang_cc1 -triple=spirv64 -fdeclare-spirv-builtins -emit-llvm %s -o - | FileCheck %s

template <class T> void test_shuffle() {
  T v;
  unsigned int id;
  __spirv_SubgroupShuffleINTEL(v, id);
  __spirv_SubgroupShuffleXorINTEL(v, id);
  __spirv_SubgroupShuffleUpINTEL(v, v, id);
  __spirv_SubgroupShuffleDownINTEL(v, v, id);
}

template <class T> void test_shuffle_scalar_and_vector() {
  test_shuffle<T>();
  test_shuffle<T __attribute__((ext_vector_type(2)))>();
  test_shuffle<T __attribute__((ext_vector_type(3)))>();
  test_shuffle<T __attribute__((ext_vector_type(4)))>();
  test_shuffle<T __attribute__((ext_vector_type(8)))>();
  test_shuffle<T __attribute__((ext_vector_type(16)))>();
}

template <class T, class PtrTy> void test_block_write_addrspace(PtrTy ptr) {
  T v;
  __spirv_SubgroupBlockWriteINTEL(ptr, v);
  using T2 = T __attribute__((ext_vector_type(2)));
  T2 v2;
  __spirv_SubgroupBlockWriteINTEL(ptr, v2);
  using T4 = T __attribute__((ext_vector_type(4)));
  T4 v4;
  __spirv_SubgroupBlockWriteINTEL(ptr, v4);
  using T8 = T __attribute__((ext_vector_type(8)));
  T8 v8;
  __spirv_SubgroupBlockWriteINTEL(ptr, v8);
}

template <class T, class PtrTy> void test_block_write_addrspace_v16(PtrTy ptr) {
  using T16 = T __attribute__((ext_vector_type(16)));
  T16 v16;
  __spirv_SubgroupBlockWriteINTEL(ptr, v16);
}

template <class T> void test_block_write() {
  __attribute__((opencl_global)) T * gptr;
  test_block_write_addrspace<T>(gptr);
  __attribute__((opencl_local)) T * lptr;
  test_block_write_addrspace<T>(lptr);
}

template <class T> void test_block_write_v16() {
  __attribute__((opencl_global)) T * gptr;
  test_block_write_addrspace_v16<T>(gptr);
  __attribute__((opencl_local)) T * lptr;
  test_block_write_addrspace_v16<T>(lptr);
}

void test() {
  test_shuffle<_Float16>();
  test_shuffle<double>();

  test_shuffle_scalar_and_vector<unsigned char>();
  test_shuffle_scalar_and_vector<int>();
  test_shuffle_scalar_and_vector<short>();
  test_shuffle_scalar_and_vector<unsigned short>();
  test_shuffle_scalar_and_vector<unsigned int>();
  test_shuffle_scalar_and_vector<float>();

  test_block_write<unsigned char>();
  test_block_write_v16<unsigned char>();
  test_block_write<unsigned short>();
  test_block_write_v16<unsigned short>();
  test_block_write<unsigned int>();
}

// CHECK: call spir_func noundef half @_Z28__spirv_SubgroupShuffleINTELDF16_j
// CHECK: call spir_func noundef half @_Z31__spirv_SubgroupShuffleXorINTELDF16_j
// CHECK: call spir_func noundef half @_Z30__spirv_SubgroupShuffleUpINTELDF16_DF16_j
// CHECK: call spir_func noundef half @_Z32__spirv_SubgroupShuffleDownINTELDF16_DF16_j
// CHECK: call spir_func noundef double @_Z28__spirv_SubgroupShuffleINTELdj
// CHECK: call spir_func noundef double @_Z31__spirv_SubgroupShuffleXorINTELdj
// CHECK: call spir_func noundef double @_Z30__spirv_SubgroupShuffleUpINTELddj
// CHECK: call spir_func noundef double @_Z32__spirv_SubgroupShuffleDownINTELddj
// CHECK: call spir_func noundef zeroext i8 @_Z28__spirv_SubgroupShuffleINTELhj
// CHECK: call spir_func noundef zeroext i8 @_Z31__spirv_SubgroupShuffleXorINTELhj
// CHECK: call spir_func noundef zeroext i8 @_Z30__spirv_SubgroupShuffleUpINTELhhj
// CHECK: call spir_func noundef zeroext i8 @_Z32__spirv_SubgroupShuffleDownINTELhhj
// CHECK: call spir_func noundef <2 x i8> @_Z28__spirv_SubgroupShuffleINTELDv2_hj
// CHECK: call spir_func noundef <2 x i8> @_Z31__spirv_SubgroupShuffleXorINTELDv2_hj
// CHECK: call spir_func noundef <2 x i8> @_Z30__spirv_SubgroupShuffleUpINTELDv2_hS_j
// CHECK: call spir_func noundef <2 x i8> @_Z32__spirv_SubgroupShuffleDownINTELDv2_hS_j
// CHECK: call spir_func noundef <3 x i8> @_Z28__spirv_SubgroupShuffleINTELDv3_hj
// CHECK: call spir_func noundef <3 x i8> @_Z31__spirv_SubgroupShuffleXorINTELDv3_hj
// CHECK: call spir_func noundef <3 x i8> @_Z30__spirv_SubgroupShuffleUpINTELDv3_hS_j
// CHECK: call spir_func noundef <3 x i8> @_Z32__spirv_SubgroupShuffleDownINTELDv3_hS_j
// CHECK: call spir_func noundef <4 x i8> @_Z28__spirv_SubgroupShuffleINTELDv4_hj
// CHECK: call spir_func noundef <4 x i8> @_Z31__spirv_SubgroupShuffleXorINTELDv4_hj
// CHECK: call spir_func noundef <4 x i8> @_Z30__spirv_SubgroupShuffleUpINTELDv4_hS_j
// CHECK: call spir_func noundef <4 x i8> @_Z32__spirv_SubgroupShuffleDownINTELDv4_hS_j
// CHECK: call spir_func noundef <8 x i8> @_Z28__spirv_SubgroupShuffleINTELDv8_hj
// CHECK: call spir_func noundef <8 x i8> @_Z31__spirv_SubgroupShuffleXorINTELDv8_hj
// CHECK: call spir_func noundef <8 x i8> @_Z30__spirv_SubgroupShuffleUpINTELDv8_hS_j
// CHECK: call spir_func noundef <8 x i8> @_Z32__spirv_SubgroupShuffleDownINTELDv8_hS_j
// CHECK: call spir_func noundef <16 x i8> @_Z28__spirv_SubgroupShuffleINTELDv16_hj
// CHECK: call spir_func noundef <16 x i8> @_Z31__spirv_SubgroupShuffleXorINTELDv16_hj
// CHECK: call spir_func noundef <16 x i8> @_Z30__spirv_SubgroupShuffleUpINTELDv16_hS_j
// CHECK: call spir_func noundef <16 x i8> @_Z32__spirv_SubgroupShuffleDownINTELDv16_hS_j
// CHECK: call spir_func noundef i32 @_Z28__spirv_SubgroupShuffleINTELij
// CHECK: call spir_func noundef i32 @_Z31__spirv_SubgroupShuffleXorINTELij
// CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupShuffleUpINTELiij
// CHECK: call spir_func noundef i32 @_Z32__spirv_SubgroupShuffleDownINTELiij
// CHECK: call spir_func noundef <2 x i32> @_Z28__spirv_SubgroupShuffleINTELDv2_ij
// CHECK: call spir_func noundef <2 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv2_ij
// CHECK: call spir_func noundef <2 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv2_iS_j
// CHECK: call spir_func noundef <2 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv2_iS_j
// CHECK: call spir_func noundef <3 x i32> @_Z28__spirv_SubgroupShuffleINTELDv3_ij
// CHECK: call spir_func noundef <3 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv3_ij
// CHECK: call spir_func noundef <3 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv3_iS_j
// CHECK: call spir_func noundef <3 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv3_iS_j
// CHECK: call spir_func noundef <4 x i32> @_Z28__spirv_SubgroupShuffleINTELDv4_ij
// CHECK: call spir_func noundef <4 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv4_ij
// CHECK: call spir_func noundef <4 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv4_iS_j
// CHECK: call spir_func noundef <4 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv4_iS_j
// CHECK: call spir_func noundef <8 x i32> @_Z28__spirv_SubgroupShuffleINTELDv8_ij
// CHECK: call spir_func noundef <8 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv8_ij
// CHECK: call spir_func noundef <8 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv8_iS_j
// CHECK: call spir_func noundef <8 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv8_iS_j
// CHECK: call spir_func noundef <16 x i32> @_Z28__spirv_SubgroupShuffleINTELDv16_ij
// CHECK: call spir_func noundef <16 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv16_ij
// CHECK: call spir_func noundef <16 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv16_iS_j
// CHECK: call spir_func noundef <16 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv16_iS_j
// CHECK: call spir_func noundef signext i16 @_Z28__spirv_SubgroupShuffleINTELsj
// CHECK: call spir_func noundef signext i16 @_Z31__spirv_SubgroupShuffleXorINTELsj
// CHECK: call spir_func noundef signext i16 @_Z30__spirv_SubgroupShuffleUpINTELssj
// CHECK: call spir_func noundef signext i16 @_Z32__spirv_SubgroupShuffleDownINTELssj
// CHECK: call spir_func noundef <2 x i16> @_Z28__spirv_SubgroupShuffleINTELDv2_sj
// CHECK: call spir_func noundef <2 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv2_sj
// CHECK: call spir_func noundef <2 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv2_sS_j
// CHECK: call spir_func noundef <2 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv2_sS_j
// CHECK: call spir_func noundef <3 x i16> @_Z28__spirv_SubgroupShuffleINTELDv3_sj
// CHECK: call spir_func noundef <3 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv3_sj
// CHECK: call spir_func noundef <3 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv3_sS_j
// CHECK: call spir_func noundef <3 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv3_sS_j
// CHECK: call spir_func noundef <4 x i16> @_Z28__spirv_SubgroupShuffleINTELDv4_sj
// CHECK: call spir_func noundef <4 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv4_sj
// CHECK: call spir_func noundef <4 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv4_sS_j
// CHECK: call spir_func noundef <4 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv4_sS_j
// CHECK: call spir_func noundef <8 x i16> @_Z28__spirv_SubgroupShuffleINTELDv8_sj
// CHECK: call spir_func noundef <8 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv8_sj
// CHECK: call spir_func noundef <8 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv8_sS_j
// CHECK: call spir_func noundef <8 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv8_sS_j
// CHECK: call spir_func noundef <16 x i16> @_Z28__spirv_SubgroupShuffleINTELDv16_sj
// CHECK: call spir_func noundef <16 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv16_sj
// CHECK: call spir_func noundef <16 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv16_sS_j
// CHECK: call spir_func noundef <16 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv16_sS_j
// CHECK: call spir_func noundef zeroext i16 @_Z28__spirv_SubgroupShuffleINTELtj
// CHECK: call spir_func noundef zeroext i16 @_Z31__spirv_SubgroupShuffleXorINTELtj
// CHECK: call spir_func noundef zeroext i16 @_Z30__spirv_SubgroupShuffleUpINTELttj
// CHECK: call spir_func noundef zeroext i16 @_Z32__spirv_SubgroupShuffleDownINTELttj
// CHECK: call spir_func noundef <2 x i16> @_Z28__spirv_SubgroupShuffleINTELDv2_tj
// CHECK: call spir_func noundef <2 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv2_tj
// CHECK: call spir_func noundef <2 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv2_tS_j
// CHECK: call spir_func noundef <2 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv2_tS_j
// CHECK: call spir_func noundef <3 x i16> @_Z28__spirv_SubgroupShuffleINTELDv3_tj
// CHECK: call spir_func noundef <3 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv3_tj
// CHECK: call spir_func noundef <3 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv3_tS_j
// CHECK: call spir_func noundef <3 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv3_tS_j
// CHECK: call spir_func noundef <4 x i16> @_Z28__spirv_SubgroupShuffleINTELDv4_tj
// CHECK: call spir_func noundef <4 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv4_tj
// CHECK: call spir_func noundef <4 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv4_tS_j
// CHECK: call spir_func noundef <4 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv4_tS_j
// CHECK: call spir_func noundef <8 x i16> @_Z28__spirv_SubgroupShuffleINTELDv8_tj
// CHECK: call spir_func noundef <8 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv8_tj
// CHECK: call spir_func noundef <8 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv8_tS_j
// CHECK: call spir_func noundef <8 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv8_tS_j
// CHECK: call spir_func noundef <16 x i16> @_Z28__spirv_SubgroupShuffleINTELDv16_tj
// CHECK: call spir_func noundef <16 x i16> @_Z31__spirv_SubgroupShuffleXorINTELDv16_tj
// CHECK: call spir_func noundef <16 x i16> @_Z30__spirv_SubgroupShuffleUpINTELDv16_tS_j
// CHECK: call spir_func noundef <16 x i16> @_Z32__spirv_SubgroupShuffleDownINTELDv16_tS_j
// CHECK: call spir_func noundef i32 @_Z28__spirv_SubgroupShuffleINTELjj
// CHECK: call spir_func noundef i32 @_Z31__spirv_SubgroupShuffleXorINTELjj
// CHECK: call spir_func noundef i32 @_Z30__spirv_SubgroupShuffleUpINTELjjj
// CHECK: call spir_func noundef i32 @_Z32__spirv_SubgroupShuffleDownINTELjjj
// CHECK: call spir_func noundef <2 x i32> @_Z28__spirv_SubgroupShuffleINTELDv2_jj
// CHECK: call spir_func noundef <2 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv2_jj
// CHECK: call spir_func noundef <2 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv2_jS_j
// CHECK: call spir_func noundef <2 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv2_jS_j
// CHECK: call spir_func noundef <3 x i32> @_Z28__spirv_SubgroupShuffleINTELDv3_jj
// CHECK: call spir_func noundef <3 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv3_jj
// CHECK: call spir_func noundef <3 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv3_jS_j
// CHECK: call spir_func noundef <3 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv3_jS_j
// CHECK: call spir_func noundef <4 x i32> @_Z28__spirv_SubgroupShuffleINTELDv4_jj
// CHECK: call spir_func noundef <4 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv4_jj
// CHECK: call spir_func noundef <4 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv4_jS_j
// CHECK: call spir_func noundef <4 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv4_jS_j
// CHECK: call spir_func noundef <8 x i32> @_Z28__spirv_SubgroupShuffleINTELDv8_jj
// CHECK: call spir_func noundef <8 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv8_jj
// CHECK: call spir_func noundef <8 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv8_jS_j
// CHECK: call spir_func noundef <8 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv8_jS_j
// CHECK: call spir_func noundef <16 x i32> @_Z28__spirv_SubgroupShuffleINTELDv16_jj
// CHECK: call spir_func noundef <16 x i32> @_Z31__spirv_SubgroupShuffleXorINTELDv16_jj
// CHECK: call spir_func noundef <16 x i32> @_Z30__spirv_SubgroupShuffleUpINTELDv16_jS_j
// CHECK: call spir_func noundef <16 x i32> @_Z32__spirv_SubgroupShuffleDownINTELDv16_jS_j
// CHECK: call spir_func noundef float @_Z28__spirv_SubgroupShuffleINTELfj
// CHECK: call spir_func noundef float @_Z31__spirv_SubgroupShuffleXorINTELfj
// CHECK: call spir_func noundef float @_Z30__spirv_SubgroupShuffleUpINTELffj
// CHECK: call spir_func noundef float @_Z32__spirv_SubgroupShuffleDownINTELffj
// CHECK: call spir_func noundef <2 x float> @_Z28__spirv_SubgroupShuffleINTELDv2_fj
// CHECK: call spir_func noundef <2 x float> @_Z31__spirv_SubgroupShuffleXorINTELDv2_fj
// CHECK: call spir_func noundef <2 x float> @_Z30__spirv_SubgroupShuffleUpINTELDv2_fS_j
// CHECK: call spir_func noundef <2 x float> @_Z32__spirv_SubgroupShuffleDownINTELDv2_fS_j
// CHECK: call spir_func noundef <3 x float> @_Z28__spirv_SubgroupShuffleINTELDv3_fj
// CHECK: call spir_func noundef <3 x float> @_Z31__spirv_SubgroupShuffleXorINTELDv3_fj
// CHECK: call spir_func noundef <3 x float> @_Z30__spirv_SubgroupShuffleUpINTELDv3_fS_j
// CHECK: call spir_func noundef <3 x float> @_Z32__spirv_SubgroupShuffleDownINTELDv3_fS_j
// CHECK: call spir_func noundef <4 x float> @_Z28__spirv_SubgroupShuffleINTELDv4_fj
// CHECK: call spir_func noundef <4 x float> @_Z31__spirv_SubgroupShuffleXorINTELDv4_fj
// CHECK: call spir_func noundef <4 x float> @_Z30__spirv_SubgroupShuffleUpINTELDv4_fS_j
// CHECK: call spir_func noundef <4 x float> @_Z32__spirv_SubgroupShuffleDownINTELDv4_fS_j
// CHECK: call spir_func noundef <8 x float> @_Z28__spirv_SubgroupShuffleINTELDv8_fj
// CHECK: call spir_func noundef <8 x float> @_Z31__spirv_SubgroupShuffleXorINTELDv8_fj
// CHECK: call spir_func noundef <8 x float> @_Z30__spirv_SubgroupShuffleUpINTELDv8_fS_j
// CHECK: call spir_func noundef <8 x float> @_Z32__spirv_SubgroupShuffleDownINTELDv8_fS_j
// CHECK: call spir_func noundef <16 x float> @_Z28__spirv_SubgroupShuffleINTELDv16_fj
// CHECK: call spir_func noundef <16 x float> @_Z31__spirv_SubgroupShuffleXorINTELDv16_fj
// CHECK: call spir_func noundef <16 x float> @_Z30__spirv_SubgroupShuffleUpINTELDv16_fS_j
// CHECK: call spir_func noundef <16 x float> @_Z32__spirv_SubgroupShuffleDownINTELDv16_fS_j
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1hh
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1hDv2_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1hDv4_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1hDv8_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3hh
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3hDv2_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3hDv4_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3hDv8_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1hDv16_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3hDv16_h
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1tt
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1tDv2_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1tDv4_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1tDv8_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3tt
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3tDv2_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3tDv4_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3tDv8_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1tDv16_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3tDv16_t
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1jj
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1jDv2_j
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1jDv4_j
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS1jDv8_j
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3jj
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3jDv2_j
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3jDv4_j
// CHECK: call spir_func void @_Z31__spirv_SubgroupBlockWriteINTELPU3AS3jDv8_j
