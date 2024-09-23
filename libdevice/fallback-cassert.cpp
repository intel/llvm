//==--- fallback-cassert.cpp - device agnostic implementation of C assert --==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "atomic.hpp"
#include "include/assert-happened.hpp"
#include "wrapper.h"

#if defined(__SPIR__) || defined(__SPIRV__)

#define ASSERT_NONE 0
#define ASSERT_START 1
#define ASSERT_FINISH 2

// definition
SPIR_GLOBAL AssertHappened SPIR_AssertHappenedMem;

DEVICE_EXTERN_C void __devicelib_assert_read(void *_Dst) {
  AssertHappened *Dst = (AssertHappened *)_Dst;
  int Flag = atomicLoad(&SPIR_AssertHappenedMem.Flag);

  if (ASSERT_NONE == Flag) {
    Dst->Flag = Flag;
    return;
  }

  if (Flag != ASSERT_FINISH)
    while (ASSERT_START == atomicLoad(&SPIR_AssertHappenedMem.Flag))
      ;

  *Dst = SPIR_AssertHappenedMem;
}

DEVICE_EXTERN_C void __devicelib_assert_fail(const char *expr, const char *file,
                                             int32_t line, const char *func,
                                             uint64_t gid0, uint64_t gid1,
                                             uint64_t gid2, uint64_t lid0,
                                             uint64_t lid1, uint64_t lid2) {
  int Expected = ASSERT_NONE;
  int Desired = ASSERT_START;

  if (atomicCompareAndSet(&SPIR_AssertHappenedMem.Flag, Desired, Expected) ==
      Expected) {
    SPIR_AssertHappenedMem.Line = line;
    SPIR_AssertHappenedMem.GID0 = gid0;
    SPIR_AssertHappenedMem.GID1 = gid1;
    SPIR_AssertHappenedMem.GID2 = gid2;
    SPIR_AssertHappenedMem.LID0 = lid0;
    SPIR_AssertHappenedMem.LID1 = lid1;
    SPIR_AssertHappenedMem.LID2 = lid2;

    int ExprLength = 0;
    int FileLength = 0;
    int FuncLength = 0;

    if (expr)
      for (const char *C = expr; *C != '\0'; ++C, ++ExprLength)
        ;
    if (file)
      for (const char *C = file; *C != '\0'; ++C, ++FileLength)
        ;
    if (func)
      for (const char *C = func; *C != '\0'; ++C, ++FuncLength)
        ;

    int MaxExprIdx = sizeof(SPIR_AssertHappenedMem.Expr) - 1;
    int MaxFileIdx = sizeof(SPIR_AssertHappenedMem.File) - 1;
    int MaxFuncIdx = sizeof(SPIR_AssertHappenedMem.Func) - 1;

    if (ExprLength < MaxExprIdx)
      MaxExprIdx = ExprLength;
    if (FileLength < MaxFileIdx)
      MaxFileIdx = FileLength;
    if (FuncLength < MaxFuncIdx)
      MaxFuncIdx = FuncLength;

    for (int Idx = 0; Idx < MaxExprIdx; ++Idx)
      SPIR_AssertHappenedMem.Expr[Idx] = expr[Idx];
    SPIR_AssertHappenedMem.Expr[MaxExprIdx] = '\0';

    for (int Idx = 0; Idx < MaxFileIdx; ++Idx)
      SPIR_AssertHappenedMem.File[Idx] = file[Idx];
    SPIR_AssertHappenedMem.File[MaxFileIdx] = '\0';

    for (int Idx = 0; Idx < MaxFuncIdx; ++Idx)
      SPIR_AssertHappenedMem.Func[Idx] = func[Idx];
    SPIR_AssertHappenedMem.Func[MaxFuncIdx] = '\0';

    // Show we've done copying
    atomicStore(&SPIR_AssertHappenedMem.Flag, ASSERT_FINISH);
  }

  // FIXME: call SPIR-V unreachable instead
  // volatile int *die = (int *)0x0;
  // *die = 0xdead;
}
#endif // __SPIR__ || __SPIRV__

#ifdef __NVPTX__

DEVICE_EXTERN_C void __assertfail(const char *__message, const char *__file,
                                  unsigned __line, const char *__function,
                                  size_t charSize);

DEVICE_EXTERN_C void __devicelib_assert_fail(const char *expr, const char *file,
                                             int32_t line, const char *func,
                                             uint64_t gid0, uint64_t gid1,
                                             uint64_t gid2, uint64_t lid0,
                                             uint64_t lid1, uint64_t lid2) {
  __assertfail(expr, file, line, func, 1);
}

DEVICE_EXTERN_C void _wassert(const char *_Message, const char *_File,
                              unsigned _Line) {
  __assertfail(_Message, _File, _Line, 0, 1);
}

#endif
