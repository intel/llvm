/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * SPDX-License-Identifier: MIT
 *
 */
#ifndef UR_ADAPTER_REGISTRY_HPP
#define UR_ADAPTER_REGISTRY_HPP 1

#include <array>

#include "logger/ur_logger.hpp"
#include "ur_util.hpp"

namespace loader {

class AdapterRegistry {
  public:
    std::vector<std::string> discoveredPlatforms;

    AdapterRegistry() {
        std::optional<std::string> altPlatforms;

        // UR_ADAPTERS_FORCE_LOAD  is for development/debug only
        try {
            altPlatforms = ur_getenv("UR_ADAPTERS_FORCE_LOAD");
        } catch (const std::invalid_argument &e) {
            logger::error(e.what());
        }
        if (!altPlatforms) {
            discoverKnownPlatforms();
        }

        std::stringstream ss(*altPlatforms);
        while (ss.good()) {
            std::string substr;
            getline(ss, substr, ',');
            discoveredPlatforms.emplace_back(substr);
        }
    }

  private:
    static constexpr std::array<const char *, 1> knownPlatformNames{
        MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0")};

    void discoverKnownPlatforms() {
        for (const auto &path : knownPlatformNames) {
            discoveredPlatforms.emplace_back(path);
        }
    }
};

} // namespace loader

#endif // UR_ADAPTER_REGISTRY_HPP
