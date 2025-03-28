// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fdeclare-spirv-builtins -emit-llvm %s -o - | FileCheck %s

bool group_elect() { return __spirv_GroupNonUniformElect(2); }

bool group_all(bool predicate) {
  return __spirv_GroupNonUniformAll(2, predicate);
}

bool group_any(bool predicate) {
  return __spirv_GroupNonUniformAny(2, predicate);
}

template <class T> bool group_all_equal(T v) {
  return __spirv_GroupNonUniformAllEqual(2, v);
}

template <class T> T group_broad_cast(T v, unsigned int id) {
  return __spirv_GroupNonUniformBroadcast(2, v, id);
}

template <class T> T group_broad_cast_first(T v) {
  return __spirv_GroupNonUniformBroadcastFirst(2, v);
}

typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

uint4 group_ballot(char v) { return __spirv_GroupNonUniformBallot(2, false); }

bool group_inverse_ballot(uint4 v) {
  return __spirv_GroupNonUniformInverseBallot(2, v);
}

bool group_ballot_bit_extract(uint4 v, unsigned int id) {
  return __spirv_GroupNonUniformBallotBitExtract(2, v, id);
}

unsigned int group_ballot_bit_count(uint4 v) {
  return __spirv_GroupNonUniformBallotBitCount(2, 0, v);
}

unsigned int group_ballot_find_lsb(uint4 v) {
  return __spirv_GroupNonUniformBallotFindLSB(2, 0, v);
}

unsigned int group_ballot_find_msb(uint4 v) {
  return __spirv_GroupNonUniformBallotFindMSB(2, 0, v);
}

template <class T> T group_shuffle(T v, unsigned int id) {
  return __spirv_GroupNonUniformShuffle(2, v, id);
}

template <class T> T group_shuffle_xor(T v, unsigned int id) {
  return __spirv_GroupNonUniformShuffleXor(2, v, id);
}

template <class T> T group_shuffle_up(T v, unsigned int id) {
  return __spirv_GroupNonUniformShuffleUp(2, v, id);
}

template <class T> T group_shuffle_down(T v, unsigned int id) {
  return __spirv_GroupNonUniformShuffleDown(2, v, id);
}

template <class T> T group_iadd(T v, unsigned int id) {
  return __spirv_GroupNonUniformIAdd(2, 0, v) +
         __spirv_GroupNonUniformIAdd(2, 0, v, id);
}

template <class T> T group_imul(T v, unsigned int id) {
  return __spirv_GroupNonUniformIMul(2, 0, v) +
         __spirv_GroupNonUniformIMul(2, 0, v, id);
}

template <class T> T group_fadd(T v, unsigned int id) {
  return __spirv_GroupNonUniformFAdd(2, 1, v) +
         __spirv_GroupNonUniformFAdd(2, 1, v, id);
}

template <class T> T group_fmin(T v, unsigned int id) {
  return __spirv_GroupNonUniformFMin(2, 0, v) +
         __spirv_GroupNonUniformFMin(2, 0, v, id);
}

template <class T> T group_fmax(T v, unsigned int id) {
  return __spirv_GroupNonUniformFMax(2, 0, v) +
         __spirv_GroupNonUniformFMax(2, 0, v, id);
}

template <class T> T group_fmul(T v, unsigned int id) {
  return __spirv_GroupNonUniformFMul(2, 0, v) +
         __spirv_GroupNonUniformFMul(2, 0, v, id);
}

template <class T> T group_umin(T v, unsigned int id) {
  return __spirv_GroupNonUniformUMin(2, 0, v) +
         __spirv_GroupNonUniformUMin(2, 0, v, id);
}

template <class T> T group_umax(T v, unsigned int id) {
  return __spirv_GroupNonUniformUMax(2, 0, v) +
         __spirv_GroupNonUniformUMax(2, 0, v, id);
}

template <class T> T group_smin(T v, unsigned int id) {
  return __spirv_GroupNonUniformSMin(2, 0, v) +
         __spirv_GroupNonUniformSMin(2, 0, v, id);
}

template <class T> T group_smax(T v, unsigned int id) {
  return __spirv_GroupNonUniformSMax(2, 0, v) +
         __spirv_GroupNonUniformSMax(2, 0, v, id);
}

template <class T> T group_bitwise_and(T v, unsigned int id) {
  return __spirv_GroupNonUniformBitwiseAnd(2, 0, v) +
         __spirv_GroupNonUniformBitwiseAnd(2, 0, v, id);
}

template <class T> T group_bitwise_or(T v, unsigned int id) {
  return __spirv_GroupNonUniformBitwiseOr(2, 0, v) +
         __spirv_GroupNonUniformBitwiseOr(2, 0, v, id);
}

template <class T> T group_bitwise_xor(T v, unsigned int id) {
  return __spirv_GroupNonUniformBitwiseXor(2, 0, v) +
         __spirv_GroupNonUniformBitwiseXor(2, 0, v, id);
}

template <class T> bool group_logical_and(bool v, unsigned int id) {
  return __spirv_GroupNonUniformLogicalAnd(2, 0, v) +
         __spirv_GroupNonUniformLogicalAnd(2, 0, v, id);
}

template <class T> bool group_logical_or(bool v, unsigned int id) {
  return __spirv_GroupNonUniformLogicalOr(2, 0, v) +
         __spirv_GroupNonUniformLogicalOr(2, 0, v, id);
}

template <class T> bool group_logical_xor(bool v, unsigned int id) {
  return __spirv_GroupNonUniformLogicalXor(2, 0, v) +
         __spirv_GroupNonUniformLogicalXor(2, 0, v, id);
}

template <class T> void test_with_bool() {
  T v = 0;
  unsigned int id = 0;
  group_all_equal<T>(v);
  group_broad_cast<T>(v, id);
  group_broad_cast_first<T>(v);
  group_shuffle(v, id);
  group_shuffle_xor(v, id);
  group_shuffle_up(v, id);
  group_shuffle_down(v, id);
}

template <class T> void test_integer() {
  T v = 0;
  unsigned int id = 0;
  group_iadd<T>(v, id);
  group_imul<T>(v, id);
  group_bitwise_and<T>(v, id);
  group_bitwise_or<T>(v, id);
  group_bitwise_xor<T>(v, id);
}

template <class T> void test_signed() {
  T v = 0;
  unsigned int id = 0;
  group_smin<T>(v, id);
  group_smax<T>(v, id);
}

template <class T> void test_unsigned() {
  T v = 0;
  unsigned int id = 0;
  group_umin<T>(v, id);
  group_umax<T>(v, id);
}

template <class T> void test_logical() {
  T v = 0;
  unsigned int id = 0;
  group_logical_and<T>(v, id);
  group_logical_or<T>(v, id);
  group_logical_xor<T>(v, id);
}

void test() {
  test_with_bool<bool>();
  test_with_bool<char>();
  test_with_bool<unsigned char>();
  test_with_bool<short>();
  test_with_bool<unsigned short>();
  test_with_bool<int>();
  test_with_bool<unsigned int>();
  test_with_bool<_Float16>();
  test_with_bool<float>();
  test_with_bool<double>();

  test_integer<char>();
  test_integer<unsigned char>();
  test_integer<short>();
  test_integer<unsigned short>();
  test_integer<int>();
  test_integer<unsigned int>();

  test_signed<char>();
  test_signed<short>();
  test_signed<int>();

  test_unsigned<unsigned char>();
  test_unsigned<unsigned short>();
  test_unsigned<unsigned int>();

  test_logical<bool>();
}

// CHECK: call noundef zeroext i1 @_Z28__spirv_GroupNonUniformElecti
// CHECK: call noundef zeroext i1 @_Z26__spirv_GroupNonUniformAllib
// CHECK: call noundef zeroext i1 @_Z26__spirv_GroupNonUniformAnyib
// CHECK: call noundef <4 x i32> @_Z29__spirv_GroupNonUniformBallotib
// CHECK: call noundef zeroext i1 @_Z36__spirv_GroupNonUniformInverseBallotiDv4_j
// CHECK: call noundef zeroext i1 @_Z39__spirv_GroupNonUniformBallotBitExtractiDv4_jj
// CHECK: call noundef i32 @_Z37__spirv_GroupNonUniformBallotBitCountiiDv4_j
// CHECK: call noundef i32 @_Z36__spirv_GroupNonUniformBallotFindLSBiiDv4_j
// CHECK: call noundef i32 @_Z36__spirv_GroupNonUniformBallotFindMSBiiDv4_j
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualib
// CHECK: call noundef zeroext i1 @_Z32__spirv_GroupNonUniformBroadcastibj
// CHECK: call noundef zeroext i1 @_Z37__spirv_GroupNonUniformBroadcastFirstib
// CHECK: call noundef zeroext i1 @_Z30__spirv_GroupNonUniformShuffleibj
// CHECK: call noundef zeroext i1 @_Z33__spirv_GroupNonUniformShuffleXoribj
// CHECK: call noundef zeroext i1 @_Z32__spirv_GroupNonUniformShuffleUpibj
// CHECK: call noundef zeroext i1 @_Z34__spirv_GroupNonUniformShuffleDownibj
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualii
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBroadcastiij
// CHECK: call noundef i32 @_Z37__spirv_GroupNonUniformBroadcastFirstii
// CHECK: call noundef i32 @_Z30__spirv_GroupNonUniformShuffleiij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformShuffleXoriij
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformShuffleUpiij
// CHECK: call noundef i32 @_Z34__spirv_GroupNonUniformShuffleDowniij
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualih
// CHECK: call noundef zeroext i8 @_Z32__spirv_GroupNonUniformBroadcastihj
// CHECK: call noundef zeroext i8 @_Z37__spirv_GroupNonUniformBroadcastFirstih
// CHECK: call noundef zeroext i8 @_Z30__spirv_GroupNonUniformShuffleihj
// CHECK: call noundef zeroext i8 @_Z33__spirv_GroupNonUniformShuffleXorihj
// CHECK: call noundef zeroext i8 @_Z32__spirv_GroupNonUniformShuffleUpihj
// CHECK: call noundef zeroext i8 @_Z34__spirv_GroupNonUniformShuffleDownihj
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualis
// CHECK: call noundef signext i16 @_Z32__spirv_GroupNonUniformBroadcastisj
// CHECK: call noundef signext i16 @_Z37__spirv_GroupNonUniformBroadcastFirstis
// CHECK: call noundef signext i16 @_Z30__spirv_GroupNonUniformShuffleisj
// CHECK: call noundef signext i16 @_Z33__spirv_GroupNonUniformShuffleXorisj
// CHECK: call noundef signext i16 @_Z32__spirv_GroupNonUniformShuffleUpisj
// CHECK: call noundef signext i16 @_Z34__spirv_GroupNonUniformShuffleDownisj
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualit
// CHECK: call noundef zeroext i16 @_Z32__spirv_GroupNonUniformBroadcastitj
// CHECK: call noundef zeroext i16 @_Z37__spirv_GroupNonUniformBroadcastFirstit
// CHECK: call noundef zeroext i16 @_Z30__spirv_GroupNonUniformShuffleitj
// CHECK: call noundef zeroext i16 @_Z33__spirv_GroupNonUniformShuffleXoritj
// CHECK: call noundef zeroext i16 @_Z32__spirv_GroupNonUniformShuffleUpitj
// CHECK: call noundef zeroext i16 @_Z34__spirv_GroupNonUniformShuffleDownitj
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualii
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBroadcastiij
// CHECK: call noundef i32 @_Z37__spirv_GroupNonUniformBroadcastFirstii
// CHECK: call noundef i32 @_Z30__spirv_GroupNonUniformShuffleiij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformShuffleXoriij
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformShuffleUpiij
// CHECK: call noundef i32 @_Z34__spirv_GroupNonUniformShuffleDowniij
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualij
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBroadcastijj
// CHECK: call noundef i32 @_Z37__spirv_GroupNonUniformBroadcastFirstij
// CHECK: call noundef i32 @_Z30__spirv_GroupNonUniformShuffleijj
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformShuffleXorijj
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformShuffleUpijj
// CHECK: call noundef i32 @_Z34__spirv_GroupNonUniformShuffleDownijj
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualiDF16_
// CHECK: call noundef half @_Z32__spirv_GroupNonUniformBroadcastiDF16_j
// CHECK: call noundef half @_Z37__spirv_GroupNonUniformBroadcastFirstiDF16_
// CHECK: call noundef half @_Z30__spirv_GroupNonUniformShuffleiDF16_j
// CHECK: call noundef half @_Z33__spirv_GroupNonUniformShuffleXoriDF16_j
// CHECK: call noundef half @_Z32__spirv_GroupNonUniformShuffleUpiDF16_j
// CHECK: call noundef half @_Z34__spirv_GroupNonUniformShuffleDowniDF16_j
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualif
// CHECK: call noundef float @_Z32__spirv_GroupNonUniformBroadcastifj
// CHECK: call noundef float @_Z37__spirv_GroupNonUniformBroadcastFirstif
// CHECK: call noundef float @_Z30__spirv_GroupNonUniformShuffleifj
// CHECK: call noundef float @_Z33__spirv_GroupNonUniformShuffleXorifj
// CHECK: call noundef float @_Z32__spirv_GroupNonUniformShuffleUpifj
// CHECK: call noundef float @_Z34__spirv_GroupNonUniformShuffleDownifj
// CHECK: call noundef zeroext i1 @_Z31__spirv_GroupNonUniformAllEqualid
// CHECK: call noundef double @_Z32__spirv_GroupNonUniformBroadcastidj
// CHECK: call noundef double @_Z37__spirv_GroupNonUniformBroadcastFirstid
// CHECK: call noundef double @_Z30__spirv_GroupNonUniformShuffleidj
// CHECK: call noundef double @_Z33__spirv_GroupNonUniformShuffleXoridj
// CHECK: call noundef double @_Z32__spirv_GroupNonUniformShuffleUpidj
// CHECK: call noundef double @_Z34__spirv_GroupNonUniformShuffleDownidj
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIAddiii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIAddiiij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIMuliii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIMuliiij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseAndiii
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseAndiiij
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBitwiseOriii
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBitwiseOriiij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseXoriii
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseXoriiij
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformIAddiih
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformIAddiihj
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformIMuliih
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformIMuliihj
// CHECK: call noundef zeroext i8 @_Z33__spirv_GroupNonUniformBitwiseAndiih
// CHECK: call noundef zeroext i8 @_Z33__spirv_GroupNonUniformBitwiseAndiihj
// CHECK: call noundef zeroext i8 @_Z32__spirv_GroupNonUniformBitwiseOriih
// CHECK: call noundef zeroext i8 @_Z32__spirv_GroupNonUniformBitwiseOriihj
// CHECK: call noundef zeroext i8 @_Z33__spirv_GroupNonUniformBitwiseXoriih
// CHECK: call noundef zeroext i8 @_Z33__spirv_GroupNonUniformBitwiseXoriihj
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformIAddiis
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformIAddiisj
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformIMuliis
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformIMuliisj
// CHECK: call noundef signext i16 @_Z33__spirv_GroupNonUniformBitwiseAndiis
// CHECK: call noundef signext i16 @_Z33__spirv_GroupNonUniformBitwiseAndiisj
// CHECK: call noundef signext i16 @_Z32__spirv_GroupNonUniformBitwiseOriis
// CHECK: call noundef signext i16 @_Z32__spirv_GroupNonUniformBitwiseOriisj
// CHECK: call noundef signext i16 @_Z33__spirv_GroupNonUniformBitwiseXoriis
// CHECK: call noundef signext i16 @_Z33__spirv_GroupNonUniformBitwiseXoriisj
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformIAddiit
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformIAddiitj
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformIMuliit
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformIMuliitj
// CHECK: call noundef zeroext i16 @_Z33__spirv_GroupNonUniformBitwiseAndiit
// CHECK: call noundef zeroext i16 @_Z33__spirv_GroupNonUniformBitwiseAndiitj
// CHECK: call noundef zeroext i16 @_Z32__spirv_GroupNonUniformBitwiseOriit
// CHECK: call noundef zeroext i16 @_Z32__spirv_GroupNonUniformBitwiseOriitj
// CHECK: call noundef zeroext i16 @_Z33__spirv_GroupNonUniformBitwiseXoriit
// CHECK: call noundef zeroext i16 @_Z33__spirv_GroupNonUniformBitwiseXoriitj
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIAddiii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIAddiiij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIMuliii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIMuliiij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseAndiii
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseAndiiij
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBitwiseOriii
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBitwiseOriiij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseXoriii
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseXoriiij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIAddiij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIAddiijj
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIMuliij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformIMuliijj
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseAndiij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseAndiijj
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBitwiseOriij
// CHECK: call noundef i32 @_Z32__spirv_GroupNonUniformBitwiseOriijj
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseXoriij
// CHECK: call noundef i32 @_Z33__spirv_GroupNonUniformBitwiseXoriijj
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMiniii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMiniiij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMaxiii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMaxiiij
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformSMiniis
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformSMiniisj
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformSMaxiis
// CHECK: call noundef signext i16 @_Z27__spirv_GroupNonUniformSMaxiisj
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMiniii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMiniiij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMaxiii
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformSMaxiiij
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformUMiniih
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformUMiniihj
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformUMaxiih
// CHECK: call noundef zeroext i8 @_Z27__spirv_GroupNonUniformUMaxiihj
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformUMiniit
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformUMiniitj
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformUMaxiit
// CHECK: call noundef zeroext i16 @_Z27__spirv_GroupNonUniformUMaxiitj
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformUMiniij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformUMiniijj
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformUMaxiij
// CHECK: call noundef i32 @_Z27__spirv_GroupNonUniformUMaxiijj
// CHECK: call noundef zeroext i1 @_Z33__spirv_GroupNonUniformLogicalAndiib
// CHECK: call noundef zeroext i1 @_Z33__spirv_GroupNonUniformLogicalAndiibj
// CHECK: call noundef zeroext i1 @_Z32__spirv_GroupNonUniformLogicalOriib
// CHECK: call noundef zeroext i1 @_Z32__spirv_GroupNonUniformLogicalOriibj
// CHECK: call noundef zeroext i1 @_Z33__spirv_GroupNonUniformLogicalXoriib
// CHECK: call noundef zeroext i1 @_Z33__spirv_GroupNonUniformLogicalXoriibj
