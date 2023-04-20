// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT

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
