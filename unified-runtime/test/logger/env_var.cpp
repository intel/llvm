// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

//////////////////////////////////////////////////////////////////////////////
TEST_F(LoggerFromEnvVar, DebugMessage) {
  UR_LOG(Debug, "Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, InfoMessage) {
  UR_LOG(Info, "Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, WarningMessage) {
  UR_LOG(Warning, "Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, ErrorMessage) {
  UR_LOG(Error, "Test message: {}", "success");
}
