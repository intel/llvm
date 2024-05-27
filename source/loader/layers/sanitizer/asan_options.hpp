/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_options.hpp
 *
 */

#pragma once

#include "common/ur_util.hpp"
#include "ur/ur.hpp"
#include "ur_sanitizer_layer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ur_sanitizer_layer {

struct AsanOptions {
  public:
    AsanOptions(AsanOptions &other) = delete;
    void operator=(const AsanOptions &) = delete;

    static AsanOptions &getInstance() {
        static AsanOptions instance;
        return instance;
    }

    bool Debug = false;
    uint64_t MinRZSize = 16;
    uint64_t MaxRZSize = 2048;
    uint32_t MaxQuarantineSizeMB = 0;
    bool DetectLocals = true;

  private:
    AsanOptions() {
        auto OptionsEnvMap = getenv_to_map("UR_LAYER_ASAN_OPTIONS");
        if (!OptionsEnvMap.has_value()) {
            return;
        }

        const char *TrueStrings[] = {"1", "true"};
        const char *FalseStrings[] = {"0", "false"};

        auto InplaceToLower = [](std::string &S) {
            std::transform(S.begin(), S.end(), S.begin(),
                           [](unsigned char C) { return std::tolower(C); });
        };
        auto IsTrue = [&](const std::string &S) {
            return std::any_of(std::begin(TrueStrings), std::end(TrueStrings),
                               [&](const char *CS) { return S == CS; });
        };
        auto IsFalse = [&](const std::string &S) {
            return std::any_of(std::begin(FalseStrings), std::end(FalseStrings),
                               [&](const char *CS) { return S == CS; });
        };

        auto SetBoolOption = [&](const std::string &Name, bool &Opt) {
            auto KV = OptionsEnvMap->find(Name);
            if (KV != OptionsEnvMap->end()) {
                auto Value = KV->second.front();
                InplaceToLower(Value);
                if (IsTrue(Value)) {
                    Opt = true;
                } else if (IsFalse(Value)) {
                    Opt = false;
                } else {
                    std::stringstream SS;
                    SS << "<SANITIZER>[ERROR]: \"" << Name << "\" is set to \""
                       << Value << "\", which is not an valid setting. ";
                    SS << "Acceptable input are: for enable, use:";
                    for (auto &S : TrueStrings) {
                        SS << " \"" << S << "\"";
                    }
                    SS << "; ";
                    SS << "for disable, use:";
                    for (auto &S : FalseStrings) {
                        SS << " \"" << S << "\"";
                    }
                    SS << ".";
                    die(SS.str().c_str());
                }
            }
        };

        SetBoolOption("debug", Debug);
        SetBoolOption("detect_locals", DetectLocals);

        auto KV = OptionsEnvMap->find("quarantine_size_mb");
        if (KV != OptionsEnvMap->end()) {
            auto Value = KV->second.front();
            try {
                auto temp_long = std::stoul(Value);
                if (temp_long > UINT32_MAX) {
                    throw std::out_of_range("");
                }
                MaxQuarantineSizeMB = temp_long;
            } catch (...) {
                die("<SANITIZER>[ERROR]: \"quarantine_size_mb\" should be "
                    "an positive integer that smaller than or equal to "
                    "4294967295.");
            }
        }

        KV = OptionsEnvMap->find("redzone");
        if (KV != OptionsEnvMap->end()) {
            auto Value = KV->second.front();
            try {
                MinRZSize = std::stoul(Value);
                if (MinRZSize < 16) {
                    MinRZSize = 16;
                    context.logger.warning("Trying to set redzone size to a "
                                           "value less than 16 is ignored");
                }
            } catch (...) {
                die("<SANITIZER>[ERROR]: \"redzone\" should be an integer");
            }
        }

        KV = OptionsEnvMap->find("max_redzone");
        if (KV != OptionsEnvMap->end()) {
            auto Value = KV->second.front();
            try {
                MaxRZSize = std::stoul(Value);
                if (MaxRZSize > 2048) {
                    MaxRZSize = 2048;
                    context.logger.warning(
                        "Trying to set max redzone size to a "
                        "value greater than 2048 is ignored");
                }
            } catch (...) {
                die("<SANITIZER>[ERROR]: \"max_redzone\" should be an integer");
            }
        }
    }
};

inline const AsanOptions &Options() { return AsanOptions::getInstance(); }

} // namespace ur_sanitizer_layer
