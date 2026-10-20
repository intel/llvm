// Part of the LLVM Project, under the Apache License v2.0 with LLVM
// Exceptions. See https://llvm.org/LICENSE.txt for license information.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once
#include <cstdint>

// Stage increments for the in-order execution verification test.

static constexpr uint32_t DISCARD_EVENTS_STAGE_INCREMENT = 10;
static constexpr uint32_t DISCARD_EVENTS_STAGE_2_INCREMENT =
    DISCARD_EVENTS_STAGE_INCREMENT;
static constexpr uint32_t DISCARD_EVENTS_STAGE_3_INCREMENT =
    DISCARD_EVENTS_STAGE_INCREMENT * 10;
static constexpr uint32_t DISCARD_EVENTS_STAGE_4_INCREMENT =
    DISCARD_EVENTS_STAGE_INCREMENT * 100;
static constexpr uint32_t DISCARD_EVENTS_STAGE_5_INCREMENT =
    DISCARD_EVENTS_STAGE_INCREMENT * 1000;
