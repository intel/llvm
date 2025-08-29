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
#ifndef MULTI_LLVM_LLVM_VERSION_H_INCLUDED
#define MULTI_LLVM_LLVM_VERSION_H_INCLUDED

#include <llvm/Config/llvm-config.h>

#define LLVM_VERSION_EQUAL(MAJOR, MINOR) \
  (LLVM_VERSION_MAJOR == (MAJOR) && LLVM_VERSION_MINOR == (MINOR))

#define LLVM_VERSION_LESS(MAJOR, MINOR) \
  ((LLVM_VERSION_MAJOR < (MAJOR)) ||    \
   (LLVM_VERSION_MAJOR == (MAJOR) && LLVM_VERSION_MINOR < (MINOR)))

#define LLVM_VERSION_LESS_EQUAL(MAJOR, MINOR) \
  (LLVM_VERSION_EQUAL(MAJOR, MINOR) || LLVM_VERSION_LESS(MAJOR, MINOR))

#define LLVM_VERSION_GREATER(MAJOR, MINOR) \
  ((LLVM_VERSION_MAJOR > (MAJOR)) ||       \
   (LLVM_VERSION_MAJOR == (MAJOR) && LLVM_VERSION_MINOR > (MINOR)))

#define LLVM_VERSION_GREATER_EQUAL(MAJOR, MINOR) \
  (LLVM_VERSION_EQUAL(MAJOR, MINOR) || LLVM_VERSION_GREATER(MAJOR, MINOR))

#endif  // MULTI_LLVM_LLVM_VERSION_H_INCLUDED
