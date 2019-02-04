//===-- SBInitializerOptions.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "lldb/API/SBInitializerOptions.h"
#include "lldb/Initialization/SystemInitializer.h"

using namespace lldb;
using namespace lldb_private;

SBInitializerOptions::SBInitializerOptions(const SBInitializerOptions &rhs) {
  m_opaque_up.reset(new InitializerOptions());
  *(m_opaque_up.get()) = rhs.ref();
}

const SBInitializerOptions &SBInitializerOptions::
operator=(const SBInitializerOptions &rhs) {
  if (this != &rhs) {
    this->ref() = rhs.ref();
  }
  return *this;
}

SBInitializerOptions::~SBInitializerOptions() {}

SBInitializerOptions::SBInitializerOptions() {
  m_opaque_up.reset(new InitializerOptions());
}

void SBInitializerOptions::SetCaptureReproducer(bool b) {
  m_opaque_up->reproducer_capture = b;
}

void SBInitializerOptions::SetReplayReproducer(bool b) {
  m_opaque_up->reproducer_replay = b;
}

void SBInitializerOptions::SetReproducerPath(const char *path) {
  m_opaque_up->reproducer_path = path;
}

InitializerOptions &SBInitializerOptions::ref() const {
  return *(m_opaque_up.get());
}
