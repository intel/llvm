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
    AdapterRegistry() {
        std::optional<std::string> altPlatforms;

        // UR_ADAPTERS_FORCE_LOAD  is for development/debug only
        try {
            altPlatforms = ur_getenv("UR_ADAPTERS_FORCE_LOAD");
        } catch (const std::invalid_argument &e) {
            logger::error(e.what());
        }
        if (!altPlatforms) {
            discoverKnownAdapters();
        }

        std::stringstream ss(*altPlatforms);
        while (ss.good()) {
            std::string substr;
            getline(ss, substr, ',');
            discovered_adapters.emplace_back(substr);
        }
    }

    struct Iterator {
        using value_type = const std::string;
        using pointer = value_type *;

        Iterator(pointer ptr) noexcept : current_adapter(ptr) {}

        Iterator &operator++() noexcept {
            current_adapter++;
            return *this;
        }

        Iterator operator++(int) {
            Iterator tmp = *this;
            ++(*this);
            return tmp;
        }

        bool operator!=(const Iterator &other) const noexcept {
            return this->current_adapter != other.current_adapter;
        }

        const value_type operator*() const { return *this->current_adapter; }

      private:
        pointer current_adapter;
    };

    const std::string &operator[](size_t i) const {
        return discovered_adapters[i];
    }

    bool empty() const noexcept { return discovered_adapters.size() == 0; }

    size_t size() const noexcept { return discovered_adapters.size(); }

    const Iterator begin() const noexcept {
        return Iterator(&(*discovered_adapters.cbegin()));
    }

    const Iterator end() const noexcept {
        return Iterator(&(*discovered_adapters.cbegin()) +
                        discovered_adapters.size());
    }

  private:
    std::vector<std::string> discovered_adapters;

    static constexpr std::array<const char *, 1> knownPlatformNames{
        MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0")};

    void discoverKnownAdapters() {
        for (const auto &path : knownPlatformNames) {
            discovered_adapters.emplace_back(path);
        }
    }
};

} // namespace loader

#endif // UR_ADAPTER_REGISTRY_HPP
