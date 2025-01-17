// Copyright (C) 2023 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT SPDX-License-Identifier: Apache-2.0 WITH
// LLVM-exception

#include "fixtures.hpp"

//////////////////////////////////////////////////////////////////////////////
TEST_F(LoggerFromEnvVar, DebugMessage) {
  logger::debug("Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, InfoMessage) {
  logger::info("Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, WarningMessage) {
  logger::warning("Test message: {}", "success");
}

TEST_F(LoggerFromEnvVar, ErrorMessage) {
  logger::error("Test message: {}", "success");
}
