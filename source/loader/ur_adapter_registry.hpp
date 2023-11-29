/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#ifndef UR_ADAPTER_REGISTRY_HPP
#define UR_ADAPTER_REGISTRY_HPP 1

#include <array>

#include "logger/ur_logger.hpp"
#include "ur_adapter_search.hpp"
#include "ur_util.hpp"

namespace fs = filesystem;

namespace ur_loader {

class AdapterRegistry {
  public:
    AdapterRegistry() {
        std::optional<std::vector<std::string>> forceLoadedAdaptersOpt;
        try {
            forceLoadedAdaptersOpt = getenv_to_vec("UR_ADAPTERS_FORCE_LOAD");
        } catch (const std::invalid_argument &e) {
            logger::error(e.what());
        }

        if (forceLoadedAdaptersOpt.has_value()) {
            for (const auto &s : forceLoadedAdaptersOpt.value()) {
                auto path = fs::path(s);
                bool exists = false;
                try {
                    exists = fs::exists(path);
                } catch (std::exception &e) {
                    logger::error(e.what());
                }

                if (exists) {
                    adaptersLoadPaths.emplace_back(
                        std::vector{std::move(path)});
                } else {
                    logger::warning(
                        "Detected nonexistent path {} in environmental "
                        "variable UR_ADAPTERS_FORCE_LOAD",
                        s);
                }
            }
        } else {
            discoverKnownAdapters();
        }
    }

    struct Iterator {
        using value_type = const std::vector<fs::path>;
        using pointer = value_type *;

        Iterator(pointer ptr) noexcept : currentAdapterPaths(ptr) {}

        Iterator &operator++() noexcept {
            currentAdapterPaths++;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator!=(const Iterator &other) const noexcept {
            return this->currentAdapterPaths != other.currentAdapterPaths;
        }

        const value_type operator*() const {
            return *this->currentAdapterPaths;
        }

      private:
        pointer currentAdapterPaths;
    };

    const std::vector<fs::path> &operator[](size_t i) const {
        return adaptersLoadPaths[i];
    }

    bool empty() const noexcept { return adaptersLoadPaths.size() == 0; }

    size_t size() const noexcept { return adaptersLoadPaths.size(); }

    std::vector<std::vector<fs::path>>::const_iterator begin() const noexcept {
        return adaptersLoadPaths.begin();
    }

    std::vector<std::vector<fs::path>>::const_iterator end() const noexcept {
        return adaptersLoadPaths.end();
    }

    std::vector<std::vector<fs::path>>::const_iterator cbegin() const noexcept {
        return adaptersLoadPaths.cbegin();
    }

    std::vector<std::vector<fs::path>>::const_iterator cend() const noexcept {
        return adaptersLoadPaths.cend();
    }

  private:
    // Each outer vector entry corresponds to a different adapter type.
    // Inner vector entries are sorted candidate paths used when attempting
    // to load the adapter.
    std::vector<std::vector<fs::path>> adaptersLoadPaths;

    static constexpr std::array<const char *, 4> knownAdapterNames{
        MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0"),
        MAKE_LIBRARY_NAME("ur_adapter_hip", "0"),
        MAKE_LIBRARY_NAME("ur_adapter_opencl", "0"),
        MAKE_LIBRARY_NAME("ur_adapter_cuda", "0")};

    std::optional<std::vector<fs::path>> getEnvAdapterSearchPaths() {
        std::optional<std::vector<std::string>> pathStringsOpt;
        try {
            pathStringsOpt = getenv_to_vec("UR_ADAPTERS_SEARCH_PATH");
        } catch (const std::invalid_argument &e) {
            logger::error(e.what());
            return std::nullopt;
        }

        std::vector<fs::path> paths;
        if (pathStringsOpt.has_value()) {
            for (const auto &s : pathStringsOpt.value()) {
                auto path = fs::path(s);
                if (fs::exists(path)) {
                    paths.emplace_back(path);
                } else {
                    logger::warning(
                        "Detected nonexistent path {} in environmental "
                        "variable UR_ADAPTERS_SEARCH_PATH",
                        s);
                }
            }
        }

        return paths.empty() ? std::nullopt : std::optional(paths);
    }

    void discoverKnownAdapters() {
        auto searchPathsEnvOpt = getEnvAdapterSearchPaths();
        auto loaderLibPathOpt = getLoaderLibPath();
        for (const auto &adapterName : knownAdapterNames) {
            std::vector<fs::path> loadPaths;

            // Adapter search order:
            // 1. Every path from UR_ADAPTERS_SEARCH_PATH.
            // 2. OS search paths.
            // 3. Loader library directory.
            if (searchPathsEnvOpt.has_value()) {
                for (const auto &p : searchPathsEnvOpt.value()) {
                    loadPaths.emplace_back(p / adapterName);
                }
            }

            auto adapterNamePathOpt = getAdapterNameAsPath(adapterName);
            if (adapterNamePathOpt.has_value()) {
                const auto &adapterNamePath = adapterNamePathOpt.value();
                loadPaths.emplace_back(adapterNamePath);
            }

            if (loaderLibPathOpt.has_value()) {
                const auto &loaderLibPath = loaderLibPathOpt.value();
                loadPaths.emplace_back(loaderLibPath / adapterName);
            }

            adaptersLoadPaths.emplace_back(loadPaths);
        }
    }
};

} // namespace ur_loader

#endif // UR_ADAPTER_REGISTRY_HPP
