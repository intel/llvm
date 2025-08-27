// RUN: %clang_cc1 -triple=spirv64 -fdeclare-spirv-builtins -emit-llvm %s -o - | FileCheck %s

template <bool, typename If, typename Else> struct conditional_t {
  using T = If;
};

template <typename If, typename Else> struct conditional_t<false, If, Else> {
  using T = Else;
};

#define DEFINE_TEST_CONVERT_N(Op, N, DstTy, DstTyName, SrcTy, SrcTyName)       \
  namespace Op##DstTyName##SrcTyName {                                         \
    using DTy##N =                                                             \
        conditional_t<N == 1, DstTy,                                           \
                      DstTy __attribute__((ext_vector_type(N)))>::T;           \
    using STy##N =                                                             \
        conditional_t<N == 1, SrcTy,                                           \
                      SrcTy __attribute__((ext_vector_type(N)))>::T;           \
    DTy##N test_##Name##DstTyName(STy##N v) {                                  \
      return __spirv_##Op##Convert_R##DstTyName(v);                            \
    }                                                                          \
    STy##N test_##Name##SrcTyName(DTy##N v) {                                  \
      return __spirv_##Op##Convert_R##SrcTyName(v);                            \
    }                                                                          \
  }

#define DEFINE_TEST_CONVERT(Op, DstTy, DstTySpvName, SrcTy, SrcTySpvName)      \
  DEFINE_TEST_CONVERT_N(Op, 1, DstTy, DstTySpvName, SrcTy, SrcTySpvName)       \
  DEFINE_TEST_CONVERT_N(Op, 2, DstTy, DstTySpvName##2, SrcTy, SrcTySpvName##2) \
  DEFINE_TEST_CONVERT_N(Op, 3, DstTy, DstTySpvName##3, SrcTy, SrcTySpvName##3) \
  DEFINE_TEST_CONVERT_N(Op, 4, DstTy, DstTySpvName##4, SrcTy, SrcTySpvName##4) \
  DEFINE_TEST_CONVERT_N(Op, 8, DstTy, DstTySpvName##8, SrcTy, SrcTySpvName##8) \
  DEFINE_TEST_CONVERT_N(Op, 16, DstTy, DstTySpvName##16, SrcTy,                \
                        SrcTySpvName##16)

DEFINE_TEST_CONVERT(S, char, char, short, short)
DEFINE_TEST_CONVERT(S, char, char, int, int)
DEFINE_TEST_CONVERT(S, char, char, long, long)
DEFINE_TEST_CONVERT(S, signed char, schar, int, int)
DEFINE_TEST_CONVERT(S, signed char, schar, short, short)
DEFINE_TEST_CONVERT(S, signed char, schar, long, long)
DEFINE_TEST_CONVERT(S, short, short, int, int)
DEFINE_TEST_CONVERT(S, short, short, long, long)
DEFINE_TEST_CONVERT(S, int, int, long, long)

DEFINE_TEST_CONVERT(U, unsigned char, uchar, unsigned short, ushort)
DEFINE_TEST_CONVERT(U, unsigned char, uchar, unsigned int, uint)
DEFINE_TEST_CONVERT(U, unsigned char, uchar, unsigned long, ulong)
DEFINE_TEST_CONVERT(U, unsigned short, ushort, unsigned int, uint)
DEFINE_TEST_CONVERT(U, unsigned short, ushort, unsigned long, ulong)
DEFINE_TEST_CONVERT(U, unsigned int, uint, unsigned long, ulong)

// CHECK: call {{.*}} signext i8 @_Z22__spirv_SConvert_Rchars
// CHECK: call {{.*}} signext i16 @_Z23__spirv_SConvert_Rshortc
// CHECK: call {{.*}} <2 x i8> @_Z23__spirv_SConvert_Rchar2Dv2_s
// CHECK: call {{.*}} <2 x i16> @_Z24__spirv_SConvert_Rshort2Dv2_c
// CHECK: call {{.*}} <3 x i8> @_Z23__spirv_SConvert_Rchar3Dv3_s
// CHECK: call {{.*}} <3 x i16> @_Z24__spirv_SConvert_Rshort3Dv3_c
// CHECK: call {{.*}} <4 x i8> @_Z23__spirv_SConvert_Rchar4Dv4_s
// CHECK: call {{.*}} <4 x i16> @_Z24__spirv_SConvert_Rshort4Dv4_c
// CHECK: call {{.*}} <8 x i8> @_Z23__spirv_SConvert_Rchar8Dv8_s
// CHECK: call {{.*}} <8 x i16> @_Z24__spirv_SConvert_Rshort8Dv8_c
// CHECK: call {{.*}} <16 x i8> @_Z24__spirv_SConvert_Rchar16Dv16_s
// CHECK: call {{.*}} <16 x i16> @_Z25__spirv_SConvert_Rshort16Dv16_c
// CHECK: call {{.*}} signext i8 @_Z22__spirv_SConvert_Rchari
// CHECK: call {{.*}} i32 @_Z21__spirv_SConvert_Rintc
// CHECK: call {{.*}} <2 x i8> @_Z23__spirv_SConvert_Rchar2Dv2_i
// CHECK: call {{.*}} <2 x i32> @_Z22__spirv_SConvert_Rint2Dv2_c
// CHECK: call {{.*}} <3 x i8> @_Z23__spirv_SConvert_Rchar3Dv3_i
// CHECK: call {{.*}} <3 x i32> @_Z22__spirv_SConvert_Rint3Dv3_c
// CHECK: call {{.*}} <4 x i8> @_Z23__spirv_SConvert_Rchar4Dv4_i
// CHECK: call {{.*}} <4 x i32> @_Z22__spirv_SConvert_Rint4Dv4_c
// CHECK: call {{.*}} <8 x i8> @_Z23__spirv_SConvert_Rchar8Dv8_i
// CHECK: call {{.*}} <8 x i32> @_Z22__spirv_SConvert_Rint8Dv8_c
// CHECK: call {{.*}} <16 x i8> @_Z24__spirv_SConvert_Rchar16Dv16_i
// CHECK: call {{.*}} <16 x i32> @_Z23__spirv_SConvert_Rint16Dv16_c
// CHECK: call {{.*}} signext i8 @_Z22__spirv_SConvert_Rcharl
// CHECK: call {{.*}} i64 @_Z22__spirv_SConvert_Rlongc
// CHECK: call {{.*}} <2 x i8> @_Z23__spirv_SConvert_Rchar2Dv2_l
// CHECK: call {{.*}} <2 x i64> @_Z23__spirv_SConvert_Rlong2Dv2_c
// CHECK: call {{.*}} <3 x i8> @_Z23__spirv_SConvert_Rchar3Dv3_l
// CHECK: call {{.*}} <3 x i64> @_Z23__spirv_SConvert_Rlong3Dv3_c
// CHECK: call {{.*}} <4 x i8> @_Z23__spirv_SConvert_Rchar4Dv4_l
// CHECK: call {{.*}} <4 x i64> @_Z23__spirv_SConvert_Rlong4Dv4_c
// CHECK: call {{.*}} <8 x i8> @_Z23__spirv_SConvert_Rchar8Dv8_l
// CHECK: call {{.*}} <8 x i64> @_Z23__spirv_SConvert_Rlong8Dv8_c
// CHECK: call {{.*}} <16 x i8> @_Z24__spirv_SConvert_Rchar16Dv16_l
// CHECK: call {{.*}} <16 x i64> @_Z24__spirv_SConvert_Rlong16Dv16_c
// CHECK: call {{.*}} signext i8 @_Z23__spirv_SConvert_Rschari
// CHECK: call {{.*}} i32 @_Z21__spirv_SConvert_Rinta
// CHECK: call {{.*}} <2 x i8> @_Z24__spirv_SConvert_Rschar2Dv2_i
// CHECK: call {{.*}} <2 x i32> @_Z22__spirv_SConvert_Rint2Dv2_a
// CHECK: call {{.*}} <3 x i8> @_Z24__spirv_SConvert_Rschar3Dv3_i
// CHECK: call {{.*}} <3 x i32> @_Z22__spirv_SConvert_Rint3Dv3_a
// CHECK: call {{.*}} <4 x i8> @_Z24__spirv_SConvert_Rschar4Dv4_i
// CHECK: call {{.*}} <4 x i32> @_Z22__spirv_SConvert_Rint4Dv4_a
// CHECK: call {{.*}} <8 x i8> @_Z24__spirv_SConvert_Rschar8Dv8_i
// CHECK: call {{.*}} <8 x i32> @_Z22__spirv_SConvert_Rint8Dv8_a
// CHECK: call {{.*}} <16 x i8> @_Z25__spirv_SConvert_Rschar16Dv16_i
// CHECK: call {{.*}} <16 x i32> @_Z23__spirv_SConvert_Rint16Dv16_a
// CHECK: call {{.*}} signext i8 @_Z23__spirv_SConvert_Rschars
// CHECK: call {{.*}} signext i16 @_Z23__spirv_SConvert_Rshorta
// CHECK: call {{.*}} <2 x i8> @_Z24__spirv_SConvert_Rschar2Dv2_s
// CHECK: call {{.*}} <2 x i16> @_Z24__spirv_SConvert_Rshort2Dv2_a
// CHECK: call {{.*}} <3 x i8> @_Z24__spirv_SConvert_Rschar3Dv3_s
// CHECK: call {{.*}} <3 x i16> @_Z24__spirv_SConvert_Rshort3Dv3_a
// CHECK: call {{.*}} <4 x i8> @_Z24__spirv_SConvert_Rschar4Dv4_s
// CHECK: call {{.*}} <4 x i16> @_Z24__spirv_SConvert_Rshort4Dv4_a
// CHECK: call {{.*}} <8 x i8> @_Z24__spirv_SConvert_Rschar8Dv8_s
// CHECK: call {{.*}} <8 x i16> @_Z24__spirv_SConvert_Rshort8Dv8_a
// CHECK: call {{.*}} <16 x i8> @_Z25__spirv_SConvert_Rschar16Dv16_s
// CHECK: call {{.*}} <16 x i16> @_Z25__spirv_SConvert_Rshort16Dv16_a
// CHECK: call {{.*}} signext i8 @_Z23__spirv_SConvert_Rscharl
// CHECK: call {{.*}} i64 @_Z22__spirv_SConvert_Rlonga
// CHECK: call {{.*}} <2 x i8> @_Z24__spirv_SConvert_Rschar2Dv2_l
// CHECK: call {{.*}} <2 x i64> @_Z23__spirv_SConvert_Rlong2Dv2_a
// CHECK: call {{.*}} <3 x i8> @_Z24__spirv_SConvert_Rschar3Dv3_l
// CHECK: call {{.*}} <3 x i64> @_Z23__spirv_SConvert_Rlong3Dv3_a
// CHECK: call {{.*}} <4 x i8> @_Z24__spirv_SConvert_Rschar4Dv4_l
// CHECK: call {{.*}} <4 x i64> @_Z23__spirv_SConvert_Rlong4Dv4_a
// CHECK: call {{.*}} <8 x i8> @_Z24__spirv_SConvert_Rschar8Dv8_l
// CHECK: call {{.*}} <8 x i64> @_Z23__spirv_SConvert_Rlong8Dv8_a
// CHECK: call {{.*}} <16 x i8> @_Z25__spirv_SConvert_Rschar16Dv16_l
// CHECK: call {{.*}} <16 x i64> @_Z24__spirv_SConvert_Rlong16Dv16_a
// CHECK: call {{.*}} signext i16 @_Z23__spirv_SConvert_Rshorti
// CHECK: call {{.*}} i32 @_Z21__spirv_SConvert_Rints
// CHECK: call {{.*}} <2 x i16> @_Z24__spirv_SConvert_Rshort2Dv2_i
// CHECK: call {{.*}} <2 x i32> @_Z22__spirv_SConvert_Rint2Dv2_s
// CHECK: call {{.*}} <3 x i16> @_Z24__spirv_SConvert_Rshort3Dv3_i
// CHECK: call {{.*}} <3 x i32> @_Z22__spirv_SConvert_Rint3Dv3_s
// CHECK: call {{.*}} <4 x i16> @_Z24__spirv_SConvert_Rshort4Dv4_i
// CHECK: call {{.*}} <4 x i32> @_Z22__spirv_SConvert_Rint4Dv4_s
// CHECK: call {{.*}} <8 x i16> @_Z24__spirv_SConvert_Rshort8Dv8_i
// CHECK: call {{.*}} <8 x i32> @_Z22__spirv_SConvert_Rint8Dv8_s
// CHECK: call {{.*}} <16 x i16> @_Z25__spirv_SConvert_Rshort16Dv16_i
// CHECK: call {{.*}} <16 x i32> @_Z23__spirv_SConvert_Rint16Dv16_s
// CHECK: call {{.*}} signext i16 @_Z23__spirv_SConvert_Rshortl
// CHECK: call {{.*}} i64 @_Z22__spirv_SConvert_Rlongs
// CHECK: call {{.*}} <2 x i16> @_Z24__spirv_SConvert_Rshort2Dv2_l
// CHECK: call {{.*}} <2 x i64> @_Z23__spirv_SConvert_Rlong2Dv2_s
// CHECK: call {{.*}} <3 x i16> @_Z24__spirv_SConvert_Rshort3Dv3_l
// CHECK: call {{.*}} <3 x i64> @_Z23__spirv_SConvert_Rlong3Dv3_s
// CHECK: call {{.*}} <4 x i16> @_Z24__spirv_SConvert_Rshort4Dv4_l
// CHECK: call {{.*}} <4 x i64> @_Z23__spirv_SConvert_Rlong4Dv4_s
// CHECK: call {{.*}} <8 x i16> @_Z24__spirv_SConvert_Rshort8Dv8_l
// CHECK: call {{.*}} <8 x i64> @_Z23__spirv_SConvert_Rlong8Dv8_s
// CHECK: call {{.*}} <16 x i16> @_Z25__spirv_SConvert_Rshort16Dv16_l
// CHECK: call {{.*}} <16 x i64> @_Z24__spirv_SConvert_Rlong16Dv16_s
// CHECK: call {{.*}} i32 @_Z21__spirv_SConvert_Rintl
// CHECK: call {{.*}} i64 @_Z22__spirv_SConvert_Rlongi
// CHECK: call {{.*}} <2 x i32> @_Z22__spirv_SConvert_Rint2Dv2_l
// CHECK: call {{.*}} <2 x i64> @_Z23__spirv_SConvert_Rlong2Dv2_i
// CHECK: call {{.*}} <3 x i32> @_Z22__spirv_SConvert_Rint3Dv3_l
// CHECK: call {{.*}} <3 x i64> @_Z23__spirv_SConvert_Rlong3Dv3_i
// CHECK: call {{.*}} <4 x i32> @_Z22__spirv_SConvert_Rint4Dv4_l
// CHECK: call {{.*}} <4 x i64> @_Z23__spirv_SConvert_Rlong4Dv4_i
// CHECK: call {{.*}} <8 x i32> @_Z22__spirv_SConvert_Rint8Dv8_l
// CHECK: call {{.*}} <8 x i64> @_Z23__spirv_SConvert_Rlong8Dv8_i
// CHECK: call {{.*}} <16 x i32> @_Z23__spirv_SConvert_Rint16Dv16_l
// CHECK: call {{.*}} <16 x i64> @_Z24__spirv_SConvert_Rlong16Dv16_i
// CHECK: call {{.*}} zeroext i8 @_Z23__spirv_UConvert_Ruchart
// CHECK: call {{.*}} zeroext i16 @_Z24__spirv_UConvert_Rushorth
// CHECK: call {{.*}} <2 x i8> @_Z24__spirv_UConvert_Ruchar2Dv2_t
// CHECK: call {{.*}} <2 x i16> @_Z25__spirv_UConvert_Rushort2Dv2_h
// CHECK: call {{.*}} <3 x i8> @_Z24__spirv_UConvert_Ruchar3Dv3_t
// CHECK: call {{.*}} <3 x i16> @_Z25__spirv_UConvert_Rushort3Dv3_h
// CHECK: call {{.*}} <4 x i8> @_Z24__spirv_UConvert_Ruchar4Dv4_t
// CHECK: call {{.*}} <4 x i16> @_Z25__spirv_UConvert_Rushort4Dv4_h
// CHECK: call {{.*}} <8 x i8> @_Z24__spirv_UConvert_Ruchar8Dv8_t
// CHECK: call {{.*}} <8 x i16> @_Z25__spirv_UConvert_Rushort8Dv8_h
// CHECK: call {{.*}} <16 x i8> @_Z25__spirv_UConvert_Ruchar16Dv16_t
// CHECK: call {{.*}} <16 x i16> @_Z26__spirv_UConvert_Rushort16Dv16_h
// CHECK: call {{.*}} zeroext i8 @_Z23__spirv_UConvert_Rucharj
// CHECK: call {{.*}} i32 @_Z22__spirv_UConvert_Ruinth
// CHECK: call {{.*}} <2 x i8> @_Z24__spirv_UConvert_Ruchar2Dv2_j
// CHECK: call {{.*}} <2 x i32> @_Z23__spirv_UConvert_Ruint2Dv2_h
// CHECK: call {{.*}} <3 x i8> @_Z24__spirv_UConvert_Ruchar3Dv3_j
// CHECK: call {{.*}} <3 x i32> @_Z23__spirv_UConvert_Ruint3Dv3_h
// CHECK: call {{.*}} <4 x i8> @_Z24__spirv_UConvert_Ruchar4Dv4_j
// CHECK: call {{.*}} <4 x i32> @_Z23__spirv_UConvert_Ruint4Dv4_h
// CHECK: call {{.*}} <8 x i8> @_Z24__spirv_UConvert_Ruchar8Dv8_j
// CHECK: call {{.*}} <8 x i32> @_Z23__spirv_UConvert_Ruint8Dv8_h
// CHECK: call {{.*}} <16 x i8> @_Z25__spirv_UConvert_Ruchar16Dv16_j
// CHECK: call {{.*}} <16 x i32> @_Z24__spirv_UConvert_Ruint16Dv16_h
// CHECK: call {{.*}} zeroext i8 @_Z23__spirv_UConvert_Rucharm
// CHECK: call {{.*}} i64 @_Z23__spirv_UConvert_Rulongh
// CHECK: call {{.*}} <2 x i8> @_Z24__spirv_UConvert_Ruchar2Dv2_m
// CHECK: call {{.*}} <2 x i64> @_Z24__spirv_UConvert_Rulong2Dv2_h
// CHECK: call {{.*}} <3 x i8> @_Z24__spirv_UConvert_Ruchar3Dv3_m
// CHECK: call {{.*}} <3 x i64> @_Z24__spirv_UConvert_Rulong3Dv3_h
// CHECK: call {{.*}} <4 x i8> @_Z24__spirv_UConvert_Ruchar4Dv4_m
// CHECK: call {{.*}} <4 x i64> @_Z24__spirv_UConvert_Rulong4Dv4_h
// CHECK: call {{.*}} <8 x i8> @_Z24__spirv_UConvert_Ruchar8Dv8_m
// CHECK: call {{.*}} <8 x i64> @_Z24__spirv_UConvert_Rulong8Dv8_h
// CHECK: call {{.*}} <16 x i8> @_Z25__spirv_UConvert_Ruchar16Dv16_m
// CHECK: call {{.*}} <16 x i64> @_Z25__spirv_UConvert_Rulong16Dv16_h
// CHECK: call {{.*}} zeroext i16 @_Z24__spirv_UConvert_Rushortj
// CHECK: call {{.*}} i32 @_Z22__spirv_UConvert_Ruintt
// CHECK: call {{.*}} <2 x i16> @_Z25__spirv_UConvert_Rushort2Dv2_j
// CHECK: call {{.*}} <2 x i32> @_Z23__spirv_UConvert_Ruint2Dv2_t
// CHECK: call {{.*}} <3 x i16> @_Z25__spirv_UConvert_Rushort3Dv3_j
// CHECK: call {{.*}} <3 x i32> @_Z23__spirv_UConvert_Ruint3Dv3_t
// CHECK: call {{.*}} <4 x i16> @_Z25__spirv_UConvert_Rushort4Dv4_j
// CHECK: call {{.*}} <4 x i32> @_Z23__spirv_UConvert_Ruint4Dv4_t
// CHECK: call {{.*}} <8 x i16> @_Z25__spirv_UConvert_Rushort8Dv8_j
// CHECK: call {{.*}} <8 x i32> @_Z23__spirv_UConvert_Ruint8Dv8_t
// CHECK: call {{.*}} <16 x i16> @_Z26__spirv_UConvert_Rushort16Dv16_j
// CHECK: call {{.*}} <16 x i32> @_Z24__spirv_UConvert_Ruint16Dv16_t
// CHECK: call {{.*}} zeroext i16 @_Z24__spirv_UConvert_Rushortm
// CHECK: call {{.*}} i64 @_Z23__spirv_UConvert_Rulongt
// CHECK: call {{.*}} <2 x i16> @_Z25__spirv_UConvert_Rushort2Dv2_m
// CHECK: call {{.*}} <2 x i64> @_Z24__spirv_UConvert_Rulong2Dv2_t
// CHECK: call {{.*}} <3 x i16> @_Z25__spirv_UConvert_Rushort3Dv3_m
// CHECK: call {{.*}} <3 x i64> @_Z24__spirv_UConvert_Rulong3Dv3_t
// CHECK: call {{.*}} <4 x i16> @_Z25__spirv_UConvert_Rushort4Dv4_m
// CHECK: call {{.*}} <4 x i64> @_Z24__spirv_UConvert_Rulong4Dv4_t
// CHECK: call {{.*}} <8 x i16> @_Z25__spirv_UConvert_Rushort8Dv8_m
// CHECK: call {{.*}} <8 x i64> @_Z24__spirv_UConvert_Rulong8Dv8_t
// CHECK: call {{.*}} <16 x i16> @_Z26__spirv_UConvert_Rushort16Dv16_m
// CHECK: call {{.*}} <16 x i64> @_Z25__spirv_UConvert_Rulong16Dv16_t
// CHECK: call {{.*}} i32 @_Z22__spirv_UConvert_Ruintm
// CHECK: call {{.*}} i64 @_Z23__spirv_UConvert_Rulongj
// CHECK: call {{.*}} <2 x i32> @_Z23__spirv_UConvert_Ruint2Dv2_m
// CHECK: call {{.*}} <2 x i64> @_Z24__spirv_UConvert_Rulong2Dv2_j
// CHECK: call {{.*}} <3 x i32> @_Z23__spirv_UConvert_Ruint3Dv3_m
// CHECK: call {{.*}} <3 x i64> @_Z24__spirv_UConvert_Rulong3Dv3_j
// CHECK: call {{.*}} <4 x i32> @_Z23__spirv_UConvert_Ruint4Dv4_m
// CHECK: call {{.*}} <4 x i64> @_Z24__spirv_UConvert_Rulong4Dv4_j
// CHECK: call {{.*}} <8 x i32> @_Z23__spirv_UConvert_Ruint8Dv8_m
// CHECK: call {{.*}} <8 x i64> @_Z24__spirv_UConvert_Rulong8Dv8_j
// CHECK: call {{.*}} <16 x i32> @_Z24__spirv_UConvert_Ruint16Dv16_m
// CHECK: call {{.*}} <16 x i64> @_Z25__spirv_UConvert_Rulong16Dv16_j
