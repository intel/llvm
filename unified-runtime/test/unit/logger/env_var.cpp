// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "fixtures.hpp"

//////////////////////////////////////////////////////////////////////////////
TEST_F(LoggerFromEnvVar, DebugMessage) {
  URLOG(DEBUG, "Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, InfoMessage) {
  URLOG(INFO, "Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, WarningMessage) {
  URLOG(WARN, "Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, ErrorMessage) {
  URLOG(ERR, "Test message: {}", "success");
}
