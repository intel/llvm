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

  static constexpr std::array<const char *, 5> knownAdapterNames{
      MAKE_LIBRARY_NAME("ur_adapter_level_zero", "0"),
      MAKE_LIBRARY_NAME("ur_adapter_opencl", "0"),
      MAKE_LIBRARY_NAME("ur_adapter_cuda", "0"),
      MAKE_LIBRARY_NAME("ur_adapter_hip", "0"),
      MAKE_LIBRARY_NAME("ur_adapter_native_cpu", "0"),
  };

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

  ur_result_t readPreFilterODS(std::string platformBackendName) {
    // TODO: Refactor this to the common code such that both the prefilter and
    // urDeviceGetSelected use the same functionality.
    bool acceptLibrary = true;
    std::optional<EnvVarMap> odsEnvMap;
    try {
      odsEnvMap = getenv_to_map("ONEAPI_DEVICE_SELECTOR", false);

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
      logger::debug("ONEAPI_DEVICE_SELECTOR Pre-Filter with backend '{}' "
                    "and platform library name '{}'",
                    backend, platformBackendName);
      enum FilterType {
        AcceptFilter,
        DiscardFilter,
      } termType = (backend.front() != '!') ? AcceptFilter : DiscardFilter;
      logger::debug(
          "termType is {}",
          (termType != AcceptFilter ? "DiscardFilter" : "AcceptFilter"));
      if (termType != AcceptFilter) {
        logger::debug("DEBUG: backend was '{}'", backend);
        backend.erase(backend.cbegin());
        logger::debug("DEBUG: backend now '{}'", backend);
      }

      // Verify that the backend string is valid, otherwise ignore the backend.
      if ((strcmp(backend.c_str(), "*") != 0) &&
          (strcmp(backend.c_str(), "level_zero") != 0) &&
          (strcmp(backend.c_str(), "opencl") != 0) &&
          (strcmp(backend.c_str(), "cuda") != 0) &&
          (strcmp(backend.c_str(), "hip") != 0)) {
        logger::debug("ONEAPI_DEVICE_SELECTOR Pre-Filter with illegal "
                      "backend '{}' ",
                      backend);
        continue;
      }

      // case-insensitive comparison by converting both tolower
      std::transform(platformBackendName.begin(), platformBackendName.end(),
                     platformBackendName.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      std::transform(backend.begin(), backend.end(), backend.begin(),
                     [](unsigned char c) { return std::tolower(c); });
      std::size_t nameFound = platformBackendName.find(backend);

      bool backendFound = nameFound != std::string::npos;
      if (termType == AcceptFilter) {
        if (backend.front() != '*' && !backendFound) {
          logger::debug("The ONEAPI_DEVICE_SELECTOR backend name '{}' was not "
                        "found in the platform library name '{}'",
                        backend, platformBackendName);
          acceptLibrary = false;
          continue;
        } else if (backend.front() == '*' || backendFound) {
          return UR_RESULT_SUCCESS;
        }
      } else {
        if (backendFound || backend.front() == '*') {
          acceptLibrary = false;
          logger::debug("The ONEAPI_DEVICE_SELECTOR backend name for discard "
                        "'{}' was found in the platform library name '{}'",
                        backend, platformBackendName);
          continue;
        }
      }
    }
    if (acceptLibrary) {
      return UR_RESULT_SUCCESS;
    }
    return UR_RESULT_ERROR_INVALID_VALUE;
  }

  void discoverKnownAdapters() {
    auto searchPathsEnvOpt = getEnvAdapterSearchPaths();
    auto loaderLibPathOpt = getLoaderLibPath();
#if defined(_WIN32)
    bool loaderPreFilter = getenv_tobool("UR_LOADER_PRELOAD_FILTER", false);
#else
    bool loaderPreFilter = getenv_tobool("UR_LOADER_PRELOAD_FILTER", true);
#endif
    for (const auto &adapterName : knownAdapterNames) {

      if (loaderPreFilter) {
        if (readPreFilterODS(adapterName) != UR_RESULT_SUCCESS) {
          logger::debug("The adapter '{}' was removed based on the "
                        "pre-filter from ONEAPI_DEVICE_SELECTOR.",
                        adapterName);
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
