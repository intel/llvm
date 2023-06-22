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

#ifndef MULTI_LLVM_OPTIONAL_HELPER_H_INCLUDED
#define MULTI_LLVM_OPTIONAL_HELPER_H_INCLUDED

#include <multi_llvm/llvm_version.h>

#if (LLVM_VERSION_MAJOR < 17)
#include <llvm/ADT/None.h>
#include <llvm/ADT/Optional.h>
#endif

#if (LLVM_VERSION_MAJOR >= 16)
#include <optional>
#endif

namespace multi_llvm {

#if (LLVM_VERSION_MAJOR >= 16)

template <typename T>
using Optional = std::optional<T>;
static constexpr std::nullopt_t None = std::nullopt;

#else

using llvm::None;
using llvm::NoneType;
template <typename T>
class Optional : public llvm::Optional<T> {
 public:
  constexpr Optional() = default;
  constexpr Optional(llvm::NoneType) {}

  constexpr Optional(const T &value) : llvm::Optional<T>(value) {}
  constexpr Optional(T &&value) : llvm::Optional<T>(std::move(value)) {}

  Optional &operator=(const T &y) {
    llvm::Optional<T>::operator=(y);
    return *this;
  }
  Optional &operator=(T &&y) {
    llvm::Optional<T>::operator=(std::forward<T>(y));
    return *this;
  }

  constexpr Optional(llvm::Optional<T> &&value)
      : llvm::Optional<T>(std::move(value)) {}

  inline constexpr bool has_value() const {
    return llvm::Optional<T>::hasValue();
  }

#if (LLVM_VERSION_MAJOR <= 14)
  inline constexpr const T &value() const {
    return llvm::Optional<T>::getValue();
  }
  inline constexpr T &value() { return llvm::Optional<T>::getValue(); }

  template <typename U>
  constexpr T value_or(U &&alt) const & {
    return llvm::Optional<T>::getValueOr(alt);
  }
#endif
};

#endif

}  // namespace multi_llvm

#endif  // MULTI_LLVM_OPTIONAL_HELPER_H_INCLUDED
