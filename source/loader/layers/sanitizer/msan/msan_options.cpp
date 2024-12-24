/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_options.cpp
 *
 */

#include "msan_options.hpp"

#include "ur/ur.hpp"
#include "ur_sanitizer_layer.hpp"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace ur_sanitizer_layer {
namespace msan {

MsanOptions::MsanOptions() {
    std::optional<EnvVarMap> OptionsEnvMap;
    try {
        OptionsEnvMap = getenv_to_map("UR_LAYER_MSAN_OPTIONS");
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
}

} // namespace msan
} // namespace ur_sanitizer_layer
