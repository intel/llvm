//==- esimd_libcm_lambda_wrapper.hpp Wrapper for calling lambda function from C
//-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef _CM_LIBCM_LAMBDA_WRAPPER_HPP_
#define _CM_LIBCM_LAMBDA_WRAPPER_HPP_

#define LAMBDA_WRAPPER_TMPL(ARGTYPE, TAG)                                      \
  typedef std::function<void(const ARGTYPE &)> LambdaFunction_##TAG;           \
  extern "C" struct LambdaWrapper_##TAG {                                      \
    LambdaFunction_##TAG f;                                                    \
    std::vector<ARGTYPE> argVector;                                            \
  };                                                                           \
                                                                               \
  template <typename LambdaTy>                                                 \
  LambdaWrapper_##TAG *makeWrapper_##TAG(std::vector<ARGTYPE> &vec,            \
                                         LambdaTy f) {                         \
    LambdaWrapper_##TAG *w = new LambdaWrapper_##TAG;                          \
    w->f = LambdaFunction_##TAG(f);                                            \
    w->argVector = vec;                                                        \
    return w;                                                                  \
  }                                                                            \
                                                                               \
  extern "C" inline void invokeLambda_##TAG(void *p) {                         \
    auto *w = reinterpret_cast<LambdaWrapper_##TAG *>(p);                      \
    w->f(w->argVector[cm_support::thread_local_idx()]);                        \
  }

#endif // _CM_LIBCM_LAMBDA_WRAPPER_HPP_
