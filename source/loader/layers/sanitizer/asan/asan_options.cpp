/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file asan_options.cpp
 *
 */

#include "asan_options.hpp"

#include "ur/ur.hpp"
#include "ur_sanitizer_layer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ur_sanitizer_layer {
namespace asan {

AsanOptions::AsanOptions() {
    std::optional<EnvVarMap> OptionsEnvMap;
    try {
        OptionsEnvMap = getenv_to_map("UR_LAYER_ASAN_OPTIONS");
    } catch (const std::invalid_argument &e) {
        std::stringstream SS;
        SS << "<SANITIZER>[ERROR]: ";
        SS << e.what();
        getContext()->logger.always(SS.str().c_str());
        die("Sanitizer failed to parse options.\n");
    }

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
                SS << "\"" << Name << "\" is set to \"" << Value
                   << "\", which is not an valid setting. ";
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
                getContext()->logger.error(SS.str().c_str());
                die("Sanitizer failed to parse options.\n");
            }
        }
    };

    SetBoolOption("debug", Debug);
    SetBoolOption("detect_kernel_arguments", DetectKernelArguments);
    SetBoolOption("detect_locals", DetectLocals);
    SetBoolOption("detect_privates", DetectPrivates);
    SetBoolOption("print_stats", PrintStats);
    SetBoolOption("detect_leaks", DetectLeaks);

    auto KV = OptionsEnvMap->find("quarantine_size_mb");
    if (KV != OptionsEnvMap->end()) {
        const auto &Value = KV->second.front();
        try {
            auto temp_long = std::stoul(Value);
            if (temp_long > UINT32_MAX) {
                throw std::out_of_range("");
            }
            MaxQuarantineSizeMB = temp_long;
        } catch (...) {
            getContext()->logger.error("\"quarantine_size_mb\" should be "
                                       "an integer in range[0, {}].",
                                       UINT32_MAX);
            die("Sanitizer failed to parse options.\n");
        }
    }

    KV = OptionsEnvMap->find("redzone");
    if (KV != OptionsEnvMap->end()) {
        const auto &Value = KV->second.front();
        try {
            MinRZSize = std::stoul(Value);
            if (MinRZSize < 16) {
                MinRZSize = 16;
                getContext()->logger.warning("Trying to set redzone size to a "
                                             "value less than 16 is ignored.");
            }
        } catch (...) {
            getContext()->logger.error(
                "\"redzone\" should be an integer in range[0, 16].");
            die("Sanitizer failed to parse options.\n");
        }
    }

    KV = OptionsEnvMap->find("max_redzone");
    if (KV != OptionsEnvMap->end()) {
        const auto &Value = KV->second.front();
        try {
            MaxRZSize = std::stoul(Value);
            if (MaxRZSize > 2048) {
                MaxRZSize = 2048;
                getContext()->logger.warning(
                    "Trying to set max redzone size to a "
                    "value greater than 2048 is ignored.");
            }
        } catch (...) {
            getContext()->logger.error(
                "\"max_redzone\" should be an integer in range[0, 2048].");
            die("Sanitizer failed to parse options.\n");
        }
    }
}

} // namespace asan
} // namespace ur_sanitizer_layer
