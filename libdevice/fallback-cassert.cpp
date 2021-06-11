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

#ifdef __SPIR__

#define ASSERT_NONE 0
#define ASSERT_START 1
#define ASSERT_FINISH 2

// definition
__SYCL_GLOBAL__ AssertHappened __SYCL_AssertHappenedMem;

static const __attribute__((opencl_constant)) char assert_fmt[] =
    "%s:%d: %s: global id: [%lu,%lu,%lu], local id: [%lu,%lu,%lu] "
    "Assertion `%s` failed.\n";

DEVICE_EXTERN_C void __devicelib_assert_read(void *_Dst) {
  AssertHappened *Dst = (AssertHappened *)_Dst;
  int Flag = Load(&__SYCL_AssertHappenedMem.Flag);

  if (ASSERT_NONE == Flag) {
    Dst->Flag = Flag;
    return;
  }
/*
  if (Flag != ASSERT_FINISH)
    while (ASSERT_START == Load(&__SYCL_AssertHappenedMem.Flag))
      ;
*/

  *Dst = __SYCL_AssertHappenedMem;
}

DEVICE_EXTERN_C void __devicelib_assert_fail(const char *expr, const char *file,
                                             int32_t line, const char *func,
                                             uint64_t gid0, uint64_t gid1,
                                             uint64_t gid2, uint64_t lid0,
                                             uint64_t lid1, uint64_t lid2) {
  // intX_t types are used instead of `int' and `long' because the format string
  // is defined in terms of *device* types (OpenCL types): %d matches a 32 bit
  // integer, %lu matches a 64 bit unsigned integer. Host `int' and
  // `long' types may be different, so we cannot use them.
  __spirv_ocl_printf(assert_fmt, file, (int32_t)line,
                     // WORKAROUND: IGC does not handle this well
                     // (func) ? func : "<unknown function>",
                     func, gid0, gid1, gid2, lid0, lid1, lid2, expr);

  {
    int Expected = ASSERT_NONE;
    int Desired = ASSERT_START;

    if (CompareAndSet(&__SYCL_AssertHappenedMem.Flag, Desired, Expected) ==
            Expected) {
      __SYCL_AssertHappenedMem.Line = line;
      __SYCL_AssertHappenedMem.GID0 = gid0;
      __SYCL_AssertHappenedMem.GID1 = gid1;
      __SYCL_AssertHappenedMem.GID2 = gid2;
      __SYCL_AssertHappenedMem.LID0 = lid0;
      __SYCL_AssertHappenedMem.LID1 = lid1;
      __SYCL_AssertHappenedMem.LID2 = lid2;

      int ExprLength = 0;
      int FileLength = 0;
      int FuncLength = 0;

      if (expr)
        for (const char *C = expr; *C != '\0'; ++C, ++ExprLength);
      if (file)
        for (const char *C = file; *C != '\0'; ++C, ++FileLength);
      if (func)
        for (const char *C = func; *C != '\0'; ++C, ++FuncLength);

      int MaxExprIdx = sizeof(__SYCL_AssertHappenedMem.Expr) - 1;
      int MaxFileIdx = sizeof(__SYCL_AssertHappenedMem.File) - 1;
      int MaxFuncIdx = sizeof(__SYCL_AssertHappenedMem.Func) - 1;

      if (ExprLength < MaxExprIdx)
        MaxExprIdx = ExprLength;
      if (FileLength < MaxFileIdx)
        MaxFileIdx = FileLength;
      if (FuncLength < MaxFuncIdx)
        MaxFuncIdx = FuncLength;

      for (int Idx = 0; Idx < MaxExprIdx; ++Idx)
        __SYCL_AssertHappenedMem.Expr[Idx] = expr[Idx];
      __SYCL_AssertHappenedMem.Expr[MaxExprIdx] = '\0';

      for (int Idx = 0; Idx < MaxFileIdx; ++Idx)
        __SYCL_AssertHappenedMem.File[Idx] = file[Idx];
      __SYCL_AssertHappenedMem.File[MaxFileIdx] = '\0';

      for (int Idx = 0; Idx < MaxFuncIdx; ++Idx)
        __SYCL_AssertHappenedMem.Func[Idx] = func[Idx];
      __SYCL_AssertHappenedMem.Func[MaxFuncIdx] = '\0';

      // Show we've done copying
      Store(&__SYCL_AssertHappenedMem.Flag, ASSERT_FINISH);
    }
  }

  // FIXME: call SPIR-V unreachable instead
  // volatile int *die = (int *)0x0;
  // *die = 0xdead;
}
#endif // __SPIR__
