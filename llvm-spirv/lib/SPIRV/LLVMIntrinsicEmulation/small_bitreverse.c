// clang-format off
/*
#include <stdlib.h>

#define  MASK2LO 0x3333333333333333LLU
#define  MASK1LO 0x5555555555555555LLU

#define  SWAP2(X,TYPE) (((X&((TYPE) MASK2LO ))<< 2) | (((X)>> 2)&((TYPE) MASK2LO)))
#define  SWAP1(X,TYPE) (((X&((TYPE) MASK1LO ))<< 1) | (((X)>> 1)&((TYPE) MASK1LO)))

#define uint2_t _BitInt(2)
#define uint4_t _BitInt(4)

///////////////////////////////////////////////////////////////////////////////////////
// scalar
///////////////////////////////////////////////////////////////////////////////////////

uint2_t llvm_bitreverse_i2(uint2_t A) {
  A=SWAP1(A,uint2_t);
  return A;
}

uint4_t llvm_bitreverse_i4(uint4_t A) {
  A=SWAP2(A,uint4_t);
  A=SWAP1(A,uint4_t);
  return A;
}
*/
// clang-format on
