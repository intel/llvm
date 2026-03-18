/*===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===*/

#include <libspirv/spirv.h>

// Assert failure format string
__constant char __assert_fmt[] = "%s:%u: %s: global id: [%u,%u,%u], local id: "
                                 "[%u,%u,%u] Assertion `%s` failed.\n";

// OCKL fprintf functions for stderr output
ulong __ockl_fprintf_stderr_begin(void);
ulong __ockl_fprintf_append_string_n(ulong msg, __constant char *str, ulong len,
                                     int is_last);
ulong __ockl_fprintf_append_args(ulong msg, int num_args, ulong arg0,
                                 ulong arg1, ulong arg2, ulong arg3, ulong arg4,
                                 ulong arg5, ulong arg6, int is_last);

__attribute__((overloadable)) size_t __spirv_BuiltInGlobalInvocationId(int);

// String length helper for assertions
ulong __strlen_assert(__constant char *str) {
  __constant char *tmp = str;
  while (*tmp != '\0') {
    tmp++;
  }
  return (ulong)(tmp - str);
}

// Assert failure handler
void __assert_fail(__constant char *assertion, __constant char *file, uint line,
                   __constant char *function) {
  // Start building the error message
  ulong msg = __ockl_fprintf_stderr_begin();

  // Append format string
  msg = __ockl_fprintf_append_string_n(
      msg, __assert_fmt, sizeof(__assert_fmt) / sizeof(char), /*is_last=*/0);

  // Append file name
  ulong len_file = __strlen_assert(file);
  msg = __ockl_fprintf_append_string_n(msg, file, len_file, 0);

  // Append line number
  msg = __ockl_fprintf_append_args(msg, /*num_args=*/1, /*arg0=*/line,
                                   /*arg1*/ 0, 0, 0, 0, 0, 0, /*is_last=*/0);

  // Append function name
  ulong len_func = __strlen_assert(function);
  msg = __ockl_fprintf_append_string_n(msg, function, len_func, /*is_last=*/0);

  // Get global invocation IDs (x, y, z)
  ulong gidx = __spirv_BuiltInGlobalInvocationId(0);
  ulong gidy = __spirv_BuiltInGlobalInvocationId(1);
  ulong gidz = __spirv_BuiltInGlobalInvocationId(2);

  // Get local invocation IDs (x, y, z)
  ulong lidx = __spirv_BuiltInLocalInvocationId(0);
  ulong lidy = __spirv_BuiltInLocalInvocationId(1);
  ulong lidz = __spirv_BuiltInLocalInvocationId(2);

  // Append all 6 ID values (global x,y,z and local x,y,z)
  msg = __ockl_fprintf_append_args(msg, 6, /*arg0*/ gidx, gidy, gidz, lidx,
                                   lidy, lidz,
                                   /*arg6*/ 0, /*is_last*/ 0);

  // Append assertion string (is_last=1)
  ulong len_assertion = __strlen_assert(assertion);
  msg = __ockl_fprintf_append_string_n(msg, assertion, len_assertion,
                                       /*is_last=*/1);

  // Trap to halt execution
  __builtin_trap();
  __builtin_unreachable();
}
