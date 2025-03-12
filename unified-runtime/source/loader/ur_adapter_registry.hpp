/*
 *
 * Copyright (C) 2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */
#ifndef UR_ADAPTER_REGISTRY_HPP
#define UR_ADAPTER_REGISTRY_HPP 1

#include <algorithm>
#include <cctype>
#include <set>

#include "logger/ur_logger.hpp"
#include "ur_adapter_search.hpp"
#include "ur_manifests.hpp"
#include "ur_util.hpp"

namespace fs = filesystem;

namespace ur_loader {

// Helper struct representing a ONEAPI_DEVICE_SELECTOR filter term.
struct FilterTerm {
  std::string backend;
  std::vector<std::string> devices;

  const std::map<std::string, ur_adapter_backend_t> backendNameMap = {
      {"opencl", UR_ADAPTER_BACKEND_OPENCL},
      {"level_zero", UR_ADAPTER_BACKEND_LEVEL_ZERO},
      {"cuda", UR_ADAPTER_BACKEND_CUDA},
      {"hip", UR_ADAPTER_BACKEND_HIP},
      {"native_cpu", UR_ADAPTER_BACKEND_NATIVE_CPU},
  };

  bool matchesBackend(const ur_adapter_manifest &manifest) const {
    if (backend.front() == '*') {
      return true;
    }

    auto backendIter = backendNameMap.find(backend);
    if (backendIter == backendNameMap.end()) {
      logger::debug(
          "ONEAPI_DEVICE_SELECTOR Pre-Filter with illegal backend '{}' ",
          backend);
      return false;
    }
    if (backendIter->second == manifest.backend) {
      return true;
    }
    return false;
  }

  const std::map<std::string, ur_device_type_t> deviceTypeMap = {
      {"cpu", UR_DEVICE_TYPE_CPU},
      {"gpu", UR_DEVICE_TYPE_GPU},
      {"fpga", UR_DEVICE_TYPE_FPGA}};

  bool matchesDevices(const ur_adapter_manifest &manifest) const {
    // If the adapter can report all device types then it matches.
    if (std::find(manifest.device_types.begin(), manifest.device_types.end(),
                  UR_DEVICE_TYPE_ALL) != manifest.device_types.end()) {
      return true;
    }
    for (auto deviceString : devices) {
      // We don't have a way to determine anything about device indices or
      // sub-devices at this stage so just match any numeric value we get.
      if (deviceString.front() == '*' || std::isdigit(deviceString.front())) {
        return true;
      }
      auto deviceIter = deviceTypeMap.find(deviceString);
      if (deviceIter == deviceTypeMap.end()) {
        logger::debug(
            "ONEAPI_DEVICE_SELECTOR Pre-Filter with illegal device '{}' ",
            deviceString);
        continue;
      }
      if (std::find(manifest.device_types.begin(), manifest.device_types.end(),
                    deviceIter->second) != manifest.device_types.end()) {
        return true;
      }
    }
    return false;
  }

  bool matches(const ur_adapter_manifest &manifest) const {
    if (!matchesBackend(manifest)) {
      return false;
    }

    return matchesDevices(manifest);
  }
};

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
        if (path.filename().extension() == STATIC_LIBRARY_EXTENSION) {
          logger::warning("UR_ADAPTERS_FORCE_LOAD contains a path to a static"
                          "library {}, it will be skipped",
                          s);
          continue;
        }

        bool exists = false;
        try {
          exists = fs::exists(path);
        } catch (std::exception &e) {
          logger::error(e.what());
        }

        if (exists) {
          forceLoaded = true;
          adaptersLoadPaths.emplace_back(std::vector{std::move(path)});
        } else {
          logger::warning("Detected nonexistent path {} in environment "
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

    const value_type operator*() const { return *this->currentAdapterPaths; }

  private:
    pointer currentAdapterPaths;
  };

  const std::vector<fs::path> &operator[](size_t i) const {
    return adaptersLoadPaths[i];
  }

  bool empty() const noexcept { return adaptersLoadPaths.size() == 0; }

  size_t size() const noexcept { return adaptersLoadPaths.size(); }

  bool adaptersForceLoaded() { return forceLoaded; }

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

  static constexpr const char *mockAdapterName =
      MAKE_LIBRARY_NAME("ur_adapter_mock", "0");

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
          logger::warning("Detected nonexistent path {} in environmental "
                          "variable UR_ADAPTERS_SEARCH_PATH",
                          s);
        }
      }
    }

    return paths.empty() ? std::nullopt : std::optional(paths);
  }

  ur_result_t getFilteredAdapterNames(std::set<std::string> &adapterNames) {
    std::optional<EnvVarMap> odsEnvMap;
    try {
      odsEnvMap = getenv_to_map("ONEAPI_DEVICE_SELECTOR", false, true);

    } catch (...) {
      // If the selector is malformed, then we ignore selector and return
      // success.
      logger::error("ERROR: missing backend, format of filter = "
                    "'[!]backend:filterStrings'");
      return UR_RESULT_SUCCESS;
    }
    logger::debug("getenv_to_map parsed env var and {} a map",
                  (odsEnvMap.has_value() ? "produced" : "failed to produce"));

    // if the ODS env var is not set at all, then pretend it was set to the
    // default
    using EnvVarMap = std::map<std::string, std::vector<std::string>>;
    EnvVarMap mapODS =
        odsEnvMap.has_value() ? odsEnvMap.value() : EnvVarMap{{"*", {"*"}}};

    std::vector<FilterTerm> positiveFilters;
    std::vector<FilterTerm> negativeFilters;

    for (auto &termPair : mapODS) {
      std::string backend = termPair.first;
      // TODO: Figure out how to process all ODS errors rather than returning
      // on the first error.
      if (backend.empty()) {
        // FIXME: never true because getenv_to_map rejects this case
        // malformed term: missing backend -- output ERROR, then continue
        logger::error("ERROR: missing backend, format of filter = "
                      "'[!]backend:filterStrings'");
        continue;
      }
      logger::debug("ONEAPI_DEVICE_SELECTOR Pre-Filter with backend '{}' ",
                    backend);

      bool PositiveFilter = backend.front() != '!';
      logger::debug("term is a {} filter",
                    (PositiveFilter ? "positive" : "negative"));
      if (!PositiveFilter) {
        logger::debug("DEBUG: backend was '{}'", backend);
        // Trim off the "!" from the backend
        backend.erase(backend.cbegin());
        logger::debug("DEBUG: backend now '{}'", backend);
      }

      // Make sure the backend is lower case
      std::transform(backend.begin(), backend.end(), backend.begin(),
                     [](unsigned char c) { return std::tolower(c); });

      if (PositiveFilter) {
        positiveFilters.push_back({backend, termPair.second});
      } else {
        // To preserve the behaviour of the original pre-filter implementation,
        // we interpret all negative filters as backend only. This isn't
        // correct, see https://github.com/intel/llvm/issues/17086
        negativeFilters.push_back({backend, {"*"}});
      }
    }

    for (const auto &manifest : ur_adapter_manifests) {
      auto matchesFilter = [manifest](const FilterTerm &f) -> bool {
        return f.matches(manifest);
      };
      if (std::any_of(positiveFilters.begin(), positiveFilters.end(),
                      matchesFilter) &&
          std::none_of(negativeFilters.begin(), negativeFilters.end(),
                       matchesFilter)) {
        adapterNames.insert(manifest.library);
      }
    }

    return UR_RESULT_SUCCESS;
  }

  void discoverKnownAdapters() {
    auto searchPathsEnvOpt = getEnvAdapterSearchPaths();
    auto loaderLibPathOpt = getLoaderLibPath();
#if defined(_WIN32)
    bool loaderPreFilter = getenv_tobool("UR_LOADER_PRELOAD_FILTER", false);
#else
    bool loaderPreFilter = getenv_tobool("UR_LOADER_PRELOAD_FILTER", true);
#endif

    std::set<std::string> adapterNames;
    if (loaderPreFilter) {
      getFilteredAdapterNames(adapterNames);
    } else {
      for (const auto &manifest : ur_adapter_manifests) {
        adapterNames.insert(manifest.library);
      }
    }

    for (const auto &adapterName : adapterNames) {
      // Skip legacy L0 adapter if the v2 adapter is requested, and vice versa.
      if (std::string(adapterName).find("level_zero") != std::string::npos) {
        auto v2Requested = getenv_tobool("UR_LOADER_USE_LEVEL_ZERO_V2", false);
        auto v2Adapter =
            std::string(adapterName).find("v2") != std::string::npos;

        if (v2Requested != v2Adapter) {
          logger::info("The adapter '{}' is skipped because {} {}.",
                       adapterName, "UR_LOADER_USE_LEVEL_ZERO_V2",
                       v2Requested ? "is set" : "is not set");
          continue;
        }
      }

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

  bool forceLoaded = false;

public:
  void enableMock() {
    adaptersLoadPaths.clear();

    std::vector<fs::path> loadPaths;
    auto adapterNamePath = fs::path{mockAdapterName};
    auto loaderLibPathOpt = getLoaderLibPath();
    if (loaderLibPathOpt.has_value()) {
      const auto &loaderLibPath = loaderLibPathOpt.value();
      loadPaths.emplace_back(loaderLibPath / adapterNamePath);
    }
    adaptersLoadPaths.emplace_back(loadPaths);
  }
};

} // namespace ur_loader

#endif // UR_ADAPTER_REGISTRY_HPP
