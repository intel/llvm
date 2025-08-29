// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/uxlfoundation/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#if LLVM == 20
template <typename T>
struct BinOpHelper<T, std::enable_if_t<T::LAST_BINOP - T::FIRST_BINOP == 18>>
#define BINOP_LLVM21(OP, STR)
#elif LLVM == 21
template <typename T, typename>
struct BinOpHelper
#define BINOP_LLVM21(OP, STR) BINOP(OP, STR)
#endif
{
#define BINOPS()                     \
  BINOP(Xchg, "xchg")                \
  BINOP(Add, "add")                  \
  BINOP(Sub, "sub")                  \
  BINOP(And, "and")                  \
  BINOP(Nand, "nand")                \
  BINOP(Or, "or")                    \
  BINOP(Xor, "xor")                  \
  BINOP(Max, "max")                  \
  BINOP(Min, "min")                  \
  BINOP(UMax, "umax")                \
  BINOP(UMin, "umin")                \
  BINOP(FAdd, "fadd")                \
  BINOP(FSub, "fsub")                \
  BINOP(FMax, "fmax")                \
  BINOP(FMin, "fmin")                \
  BINOP_LLVM21(FMaximum, "fmaximum") \
  BINOP_LLVM21(FMinimum, "fminumum") \
  BINOP(UIncWrap, "uincwrap")        \
  BINOP(UDecWrap, "udecwrap")        \
  BINOP(USubCond, "usubcond")        \
  BINOP(USubSat, "usubsat")

  static std::optional<T> consume_front_with_underscore(
      llvm::StringRef &String) {
#define BINOP(BINOP, STR)              \
  if (String.consume_front(STR "_")) { \
    return T::BINOP;                   \
  }
    BINOPS()
#undef BINOP
    return std::nullopt;
  }

  static llvm::StringRef to_string(T BinOp) {
    switch (BinOp) {
#define BINOP(BINOP, STR) \
  case T::BINOP:          \
    return STR;
      BINOPS()
#undef BINOP
      case T::BAD_BINOP:
        break;
    }
    llvm_unreachable("Unexpected BinOp");
  }

#undef BINOPS
#undef BINOP_LLVM21
};
