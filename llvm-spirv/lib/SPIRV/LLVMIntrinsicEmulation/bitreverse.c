// clang-format off
/*
#include <stdlib.h>
#include <stdint.h>

#define MASK32LO 0x00000000FFFFFFFFLLU
#define MASK16LO 0x0000FFFF0000FFFFLLU
#define  MASK8LO 0x00FF00FF00FF00FFLLU
#define  MASK4LO 0x0F0F0F0F0F0F0F0FLLU
#define  MASK2LO 0x3333333333333333LLU
#define  MASK1LO 0x5555555555555555LLU

#define SWAP32(X,TYPE) (((X&((TYPE) MASK32LO))<<32) | (((X)>>32)&((TYPE) MASK32LO)))
#define SWAP16(X,TYPE) (((X&((TYPE) MASK16LO))<<16) | (((X)>>16)&((TYPE) MASK16LO)))
#define  SWAP8(X,TYPE) (((X&((TYPE) MASK8LO ))<< 8) | (((X)>> 8)&((TYPE) MASK8LO)))
#define  SWAP4(X,TYPE) (((X&((TYPE) MASK4LO ))<< 4) | (((X)>> 4)&((TYPE) MASK4LO)))
#define  SWAP2(X,TYPE) (((X&((TYPE) MASK2LO ))<< 2) | (((X)>> 2)&((TYPE) MASK2LO)))
#define  SWAP1(X,TYPE) (((X&((TYPE) MASK1LO ))<< 1) | (((X)>> 1)&((TYPE) MASK1LO)))

///////////////////////////////////////////////////////////////////////////////////////
// scalar
///////////////////////////////////////////////////////////////////////////////////////

uint8_t llvm_bitreverse_i8(uint8_t A) {
  A=SWAP4(A,uint8_t);
  A=SWAP2(A,uint8_t);
  A=SWAP1(A,uint8_t);
  return A;
}

uint16_t llvm_bitreverse_i16(uint16_t A) {
  A=SWAP8(A,uint16_t);
  A=SWAP4(A,uint16_t);
  A=SWAP2(A,uint16_t);
  A=SWAP1(A,uint16_t);
  return A;
}

uint32_t llvm_bitreverse_i32(uint32_t A) {
  A=SWAP16(A,uint32_t);
  A=SWAP8(A,uint32_t);
  A=SWAP4(A,uint32_t);
  A=SWAP2(A,uint32_t);
  A=SWAP1(A,uint32_t);
  return A;
}

uint64_t llvm_bitreverse_i64(uint64_t A) {
  A=SWAP32(A,uint64_t);
  A=SWAP16(A,uint64_t);
  A=SWAP8(A,uint64_t);
  A=SWAP4(A,uint64_t);
  A=SWAP2(A,uint64_t);
  A=SWAP1(A,uint64_t);
  return A;
}

///////////////////////////////////////////////////////////////////////////////////////
// vector
///////////////////////////////////////////////////////////////////////////////////////

#define GEN_VECTOR_BITREVERSE(LENGTH)                                          \
typedef  uint8_t  uint8_t ## LENGTH  __attribute__((ext_vector_type(LENGTH))); \
typedef uint16_t uint16_t ## LENGTH  __attribute__((ext_vector_type(LENGTH))); \
typedef uint32_t uint32_t ## LENGTH  __attribute__((ext_vector_type(LENGTH))); \
typedef uint64_t uint64_t ## LENGTH  __attribute__((ext_vector_type(LENGTH))); \
                                                                               \
uint8_t  ## LENGTH llvm_bitreverse_v ## LENGTH ## i8 (uint8_t  ## LENGTH A) {  \
  A=SWAP4(A,uint8_t);                                                          \
  A=SWAP2(A,uint8_t);                                                          \
  A=SWAP1(A,uint8_t);                                                          \
  return A;                                                                    \
}                                                                              \
                                                                               \
uint16_t ## LENGTH llvm_bitreverse_v ## LENGTH ## i16(uint16_t ## LENGTH A) {  \
  A=SWAP8(A,uint16_t);                                                         \
  A=SWAP4(A,uint16_t);                                                         \
  A=SWAP2(A,uint16_t);                                                         \
  A=SWAP1(A,uint16_t);                                                         \
  return A;                                                                    \
}                                                                              \
                                                                               \
uint32_t ## LENGTH llvm_bitreverse_v ## LENGTH ## i32(uint32_t ## LENGTH A) {  \
  A=SWAP16(A,uint32_t);                                                        \
  A=SWAP8(A,uint32_t);                                                         \
  A=SWAP4(A,uint32_t);                                                         \
  A=SWAP2(A,uint32_t);                                                         \
  A=SWAP1(A,uint32_t);                                                         \
  return A;                                                                    \
}                                                                              \
                                                                               \
uint64_t ## LENGTH llvm_bitreverse_v ## LENGTH ## i64(uint64_t ## LENGTH A) {  \
  A=SWAP32(A,uint64_t);                                                        \
  A=SWAP16(A,uint64_t);                                                        \
  A=SWAP8(A,uint64_t);                                                         \
  A=SWAP4(A,uint64_t);                                                         \
  A=SWAP2(A,uint64_t);                                                         \
  A=SWAP1(A,uint64_t);                                                         \
  return A;                                                                    \
}

GEN_VECTOR_BITREVERSE(2)
GEN_VECTOR_BITREVERSE(3)
GEN_VECTOR_BITREVERSE(4)
GEN_VECTOR_BITREVERSE(8)
GEN_VECTOR_BITREVERSE(16)

*/
// clang-format on
