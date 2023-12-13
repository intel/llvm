// Copyright (C) Codeplay Software Limited
//
// Licensed under the Apache License, Version 2.0 (the "License") with LLVM
// Exceptions; you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://github.com/codeplaysoftware/oneapi-construction-kit/blob/main/LICENSE.txt
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/// @file
///
/// @brief Functions, macros, etc used for debugging

#ifndef VECZ_DEBUGGING_H_INCLUDED
#define VECZ_DEBUGGING_H_INCLUDED

#include <llvm/ADT/StringRef.h>
#include <llvm/IR/DiagnosticInfo.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/Instruction.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Value.h>
#include <llvm/Support/Error.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdlib>
#include <memory>
#include <optional>

namespace vecz {

/// @brief Namespace used for vecz utils that we don't want to pollute the whole
/// vecz namespace
namespace internal {
/// @brief Helper type for signaling a failure from functions that return either
/// a pointer or a boolean to indicate if vectorization was successful or not
struct VeczFailResult {
  /// @brief For functions that return a boolean value
  operator bool() const { return false; }
  /// @brief For functions that return a pointer
  template <typename T>
  operator T *() const {
    return nullptr;
  }
  /// @brief For functions that return an std::shared_ptr
  template <typename T>
  operator std::shared_ptr<T>() const {
    return nullptr;
  }
  /// @brief For functions that return an std::unique_ptr
  template <typename T>
  operator std::unique_ptr<T>() const {
    return nullptr;
  }
  /// @brief For functions that return an llvm::Optional
  template <typename T>
  operator std::optional<T>() const {
    return std::nullopt;
  }

  /// @brief For functions that return an llvm::Error
  operator llvm::Error() const {
    return llvm::make_error<llvm::StringError>("Unknown VeczFailResult",
                                               llvm::inconvertibleErrorCode());
  }
};

struct AnalysisFailResult : public internal::VeczFailResult {
  AnalysisFailResult() = default;
  ~AnalysisFailResult() = default;
  // If an optimization failed we'd better not have altered the validity of any
  // analysis...
  operator llvm::PreservedAnalyses() const {
    return llvm::PreservedAnalyses::all();
  }
};

/*
 * The following macros are available:
 *
 * VECZ_FAIL: Return from the function with a failure value (e.g. `false` or
 * `nullptr`).
 *
 * VECZ_FAIL_IF(cond): If (cond == true) then VECZ_FAIL
 *
 * VECZ_STAT_FAIL_IF(cond, stat): If (cond == true) then VECZ_FAIL and increment
 * stat
 *
 * VECZ_ERROR_IF(cond, message): Similar to VECZ_FAIL_IF, but when NDEBUG is not
 * set it aborts instead of returning a failure value.
 *
 * VECZ_ERROR(message): Similar to VECZ_ERROR_IF(true, message)
 *
 * VECZ_WARN_IF(cond, message): Similar to VECZ_ERROR_IF, but it doesn't abort
 * but warns and carries on.
 *
 * VECZ_UNREACHABLE(message): Unconditionally terminate with an error message.
 *
 * For all the macros, the message is <<'d to llvm::errs(), so it is possible to
 * print llvm Values etc. For example, this works:
 *   VECZ_WARN_IF(cond, "Warning: Value = " << *V)
 */

#define VECZ_FAIL() return vecz::internal::VeczFailResult()

#define VECZ_FAIL_IF(cond) \
  do {                     \
    if (cond) {            \
      VECZ_FAIL();         \
    }                      \
  } while (false)

#define VECZ_STAT_FAIL_IF(cond, stat) \
  do {                                \
    if (cond) {                       \
      ++stat;                         \
      VECZ_FAIL();                    \
    }                                 \
  } while (false)

#define VECZ_ERROR_IF(cond, message) \
  do {                               \
    if (cond) {                      \
      VECZ_ERROR(message);           \
    }                                \
  } while (false)

#ifdef NDEBUG

#define VECZ_ERROR(message)                                             \
  do {                                                                  \
    llvm::errs() << "!! Vecz: ERROR in " << __FILE__ << ":" << __LINE__ \
                 << "\n";                                               \
    llvm::errs() << "!! Reason: " << message << "\n";                   \
    VECZ_FAIL();                                                        \
  } while (false)

#define VECZ_WARN_IF(cond, message) /* Nothing */
#define VECZ_UNREACHABLE(message)   /* Nothing */

#else /* !NDEBUG */

#define VECZ_ERROR(message)                                             \
  do {                                                                  \
    llvm::errs() << "!! Vecz: ERROR in " << __FILE__ << ":" << __LINE__ \
                 << "\n";                                               \
    llvm::errs() << "!! Reason: " << (message) << "\n";                 \
    std::abort();                                                       \
  } while (false)

#define VECZ_WARN_IF(cond, message)                                         \
  do {                                                                      \
    if (cond) {                                                             \
      llvm::errs() << "!! Vecz: WARNING in " << __FILE__ << ":" << __LINE__ \
                   << "\n";                                                 \
      llvm::errs() << "!! Reason: " << (message) << "\n";                   \
    }                                                                       \
  } while (false)

#define VECZ_UNREACHABLE(message)                                         \
  do {                                                                    \
    llvm::errs() << "!! Vecz: UNREACHABLE reached in " << __FILE__ << ":" \
                 << __LINE__ << "\n";                                     \
    llvm::errs() << "!! Message: " << (message) << "\n";                  \
    std::abort();                                                         \
  } while (false)
#endif /* NDEBUG */
}  // namespace internal

#define VECZ_UNUSED(x) ((void)(x))

/// @brief Emit a RemarkMissed message
///
/// @param[in] F The function in which we are currently working
/// @param[in] V The value (can be `nullptr`) to be included in the message
/// @param[in] Msg The main remark message text
/// @param[in] Note An optional additional note to provide more context/info.
void emitVeczRemarkMissed(const llvm::Function *F, const llvm::Value *V,
                          llvm::StringRef Msg, llvm::StringRef Note = "");
/// @brief Emit a RemarkMissed message
///
/// @param[in] F The function in which we are currently working
/// @param[in] Msg The main remark message text
/// @param[in] Note An optional additional note to provide more context/info.
void emitVeczRemarkMissed(const llvm::Function *F, llvm::StringRef Msg,
                          llvm::StringRef Note = "");
/// @brief Emit a Remark message
///
/// @param[in] F The function in which we are currently working
/// @param[in] V The value (can be `nullptr`) to be included in the message
/// @param[in] Msg The main remark message text
void emitVeczRemark(const llvm::Function *F, const llvm::Value *V,
                    llvm::StringRef Msg);
/// @brief Emit a Remark message
///
/// @param[in] F The function in which we are currently working
/// @param[in] Msg The main remark message text
void emitVeczRemark(const llvm::Function *F, llvm::StringRef Msg);

}  // namespace vecz

#endif  // VECZ_DEBUGGING_H_INCLUDED
