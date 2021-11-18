#pragma OPENCL EXTENSION cl_khr_int64_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_int64_extended_atomics:enable

// NOTE Align these definitions with fallback-cassert.cpp
#define ASSERT_NONE 0
#define ASSERT_START 1
#define ASSERT_FINISH 2

// NOTE Layout of this structure should be aligned with the one in
// sycl/include/CL/sycl/detail/assert_happened.hpp
struct AssertHappened {
  int Flag;
  char Expr[256 + 1];
  char File[256 + 1];
  char Func[128 + 1];

  int Line;

  unsigned long GID0;
  unsigned long GID1;
  unsigned long GID2;

  unsigned long LID0;
  unsigned long LID1;
  unsigned long LID2;
};

typedef struct AssertHappened AssertHappenedT;

extern __global AssertHappenedT SPIR_AssertHappenedMem;

__kernel void __devicelib_assert_read(__global void *_Dst) {
  if (!_Dst)
    return;

  AssertHappenedT *Dst = (AssertHappenedT *)_Dst;

  __global int *FlagPtr = &SPIR_AssertHappenedMem.Flag;
  int Flag = atomic_add(FlagPtr, 0);

  if (ASSERT_NONE == Flag) {
    Dst->Flag = ASSERT_NONE;
    return;
  }

  if (Flag != ASSERT_FINISH)
    while (ASSERT_START == atomic_add(FlagPtr, 0))
      ;

  *Dst = SPIR_AssertHappenedMem;
}
