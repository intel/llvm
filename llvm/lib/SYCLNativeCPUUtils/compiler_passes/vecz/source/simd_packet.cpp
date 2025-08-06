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

#include "simd_packet.h"

#define DEBUG_TYPE "vecz-simd"

using namespace llvm;
using namespace vecz;

llvm::Value *SimdPacket::at(unsigned Index) const {
  if (Index >= size()) {
    return nullptr;
  } else {
    return (*this)[Index];
  }
}

void SimdPacket::set(unsigned Index, Value *V) {
  if (Index < size()) {
    (*this)[Index] = V;
    Mask.enable(Index);
  }
}

SimdPacket &SimdPacket::update(const SimdPacket &Other) {
  for (unsigned i = 0; i < size(); i++) {
    if (Other.Mask.isEnabled(i)) {
      (*this)[i] = Other[i];
    }
  }
  Mask.Value |= Other.Mask.Value;
  return *this;
}

void PacketMask::enableAll(unsigned NumLanes) {
  for (unsigned i = 0; i < NumLanes; i++) {
    enable(i);
  }
}
