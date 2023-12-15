/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_lib.cpp
 *
 */

// avoids windows.h from defining macros for min and max
// which avoids playing havoc with std::min and std::max
// (not quite sure why windows.h is being included here)
#define NOMINMAX

#include "ur_lib.hpp"
#include "logger/ur_logger.hpp"
#include "ur_loader.hpp"

#include <cstring> // for std::memcpy
#include <regex>

namespace ur_lib {
///////////////////////////////////////////////////////////////////////////////
context_t *context;

///////////////////////////////////////////////////////////////////////////////
context_t::context_t() {
    for (auto l : layers) {
        if (l->isAvailable()) {
            for (auto &layerName : l->getNames()) {
                availableLayers += layerName + ";";
            }
        }
    }
    // Remove the trailing ";"
    availableLayers.pop_back();
    parseEnvEnabledLayers();
}

///////////////////////////////////////////////////////////////////////////////
context_t::~context_t() {}

bool context_t::layerExists(const std::string &layerName) const {
    return availableLayers.find(layerName) != std::string::npos;
}

void context_t::parseEnvEnabledLayers() {
    auto maybeEnableEnvVarMap = getenv_to_map("UR_ENABLE_LAYERS", false);
    if (!maybeEnableEnvVarMap.has_value()) {
        return;
    }
    auto enableEnvVarMap = maybeEnableEnvVarMap.value();

    for (auto &key : enableEnvVarMap) {
        enabledLayerNames.insert(key.first);
    }
}

void context_t::initLayers() const {
    for (auto &l : layers) {
        if (l->isAvailable()) {
            l->init(&context->urDdiTable, enabledLayerNames, codelocData);
        }
    }
}

void context_t::tearDownLayers() const {
    for (auto &l : layers) {
        if (l->isAvailable()) {
            l->tearDown();
        }
    }
}

//////////////////////////////////////////////////////////////////////////
__urdlllocal ur_result_t context_t::Init(
    ur_device_init_flags_t, ur_loader_config_handle_t hLoaderConfig) {
    ur_result_t result;
    const char *logger_name = "loader";
    logger::init(logger_name);
    logger::debug("Logger {} initialized successfully!", logger_name);

    result = ur_loader::context->init();

    if (UR_RESULT_SUCCESS == result) {
        result = urLoaderInit();
    }

    if (hLoaderConfig) {
        codelocData = hLoaderConfig->codelocData;
        enabledLayerNames.merge(hLoaderConfig->getEnabledLayerNames());
    }

    if (!enabledLayerNames.empty()) {
        initLayers();
    }

    return result;
}

ur_result_t urLoaderConfigCreate(ur_loader_config_handle_t *phLoaderConfig) {
    if (!phLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    *phLoaderConfig = new ur_loader_config_handle_t_;
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigRetain(ur_loader_config_handle_t hLoaderConfig) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    hLoaderConfig->incrementReferenceCount();
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigRelease(ur_loader_config_handle_t hLoaderConfig) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    if (hLoaderConfig->decrementReferenceCount() == 0) {
        delete hLoaderConfig;
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigGetInfo(ur_loader_config_handle_t hLoaderConfig,
                                  ur_loader_config_info_t propName,
                                  size_t propSize, void *pPropValue,
                                  size_t *pPropSizeRet) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }

    if (!pPropValue && !pPropSizeRet) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    switch (propName) {
    case UR_LOADER_CONFIG_INFO_AVAILABLE_LAYERS: {
        if (pPropSizeRet) {
            *pPropSizeRet = context->availableLayers.size() + 1;
        }
        if (pPropValue) {
            char *outString = static_cast<char *>(pPropValue);
            if (propSize != context->availableLayers.size() + 1) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            std::memcpy(outString, context->availableLayers.data(),
                        propSize - 1);
            outString[propSize - 1] = '\0';
        }
        break;
    }
    case UR_LOADER_CONFIG_INFO_REFERENCE_COUNT: {
        auto refCount = hLoaderConfig->getReferenceCount();
        auto truePropSize = sizeof(refCount);
        if (pPropSizeRet) {
            *pPropSizeRet = truePropSize;
        }
        if (pPropValue) {
            if (propSize != truePropSize) {
                return UR_RESULT_ERROR_INVALID_SIZE;
            }
            std::memcpy(pPropValue, &refCount, truePropSize);
        }
        break;
    }
    default:
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
    }
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderConfigEnableLayer(ur_loader_config_handle_t hLoaderConfig,
                                      const char *pLayerName) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    if (!pLayerName) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    if (!context->layerExists(std::string(pLayerName))) {
        return UR_RESULT_ERROR_LAYER_NOT_PRESENT;
    }
    hLoaderConfig->enabledLayers.insert(pLayerName);
    return UR_RESULT_SUCCESS;
}

ur_result_t urLoaderTearDown() {
    context->tearDownLayers();

    return UR_RESULT_SUCCESS;
}

ur_result_t
urLoaderConfigSetCodeLocationCallback(ur_loader_config_handle_t hLoaderConfig,
                                      ur_code_location_callback_t pfnCodeloc,
                                      void *pUserData) {
    if (!hLoaderConfig) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    if (!pfnCodeloc) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }

    hLoaderConfig->codelocData.codelocCb = pfnCodeloc;
    hLoaderConfig->codelocData.codelocUserdata = pUserData;

    return UR_RESULT_SUCCESS;
}

ur_result_t urDeviceGetSelected(ur_platform_handle_t hPlatform,
                                ur_device_type_t DeviceType,
                                uint32_t NumEntries,
                                ur_device_handle_t *phDevices,
                                uint32_t *pNumDevices) {

    if (!hPlatform) {
        return UR_RESULT_ERROR_INVALID_NULL_HANDLE;
    }
    // NumEntries is max number of devices wanted by the caller (max usable length of phDevices)
    if (NumEntries < 0) {
        return UR_RESULT_ERROR_INVALID_SIZE;
    }
    if (NumEntries > 0 && !phDevices) {
        return UR_RESULT_ERROR_INVALID_NULL_POINTER;
    }
    // pNumDevices is the actual number of device handles added to phDevices by this function
    if (NumEntries == 0 && !pNumDevices) {
        return UR_RESULT_ERROR_INVALID_SIZE;
    }

    switch (DeviceType) {
    case UR_DEVICE_TYPE_ALL:
    case UR_DEVICE_TYPE_GPU:
    case UR_DEVICE_TYPE_DEFAULT:
    case UR_DEVICE_TYPE_CPU:
    case UR_DEVICE_TYPE_FPGA:
    case UR_DEVICE_TYPE_MCA:
        break;
    default:
        return UR_RESULT_ERROR_INVALID_ENUMERATION;
        //urPrint("Unknown device type");
        break;
    }
    // plan:
    // 0. basic validation of argument values (see code above)
    // 1. conversion of argument values into useful data items
    // 2. retrieval and parsing of environment variable string
    // 3. conversion of term map to accept and discard filters
    // 4. inserting a default "*:*" accept filter, if required
    // 5. symbolic consolidation of accept and discard filters
    // 6. querying the platform handles for all 'root' devices
    // 7. partioning via platform root devices into subdevices
    // 8. partioning via platform subdevices into subsubdevices
    // 9. short-listing devices to accept using accept filters
    // A. de-listing devices to discard using discard filters

    // possible symbolic short-circuit special cases exist:
    // * if there are no terms,     select all   root devices
    // * if any discard is "*",     select no    root devices
    // * if any discard is "*.*",   select no     sub-devices
    // * if any discard is "*.*.*", select no sub-sub-devices
    // *
    //
    // detail for step 5 of above plan:
    // * combine all accept filters into a single accept list
    // * combine all discard filters into single discard list
    // then invert it to make the initial/default accept list
    // (needs knowledge of the valid range from the platform)
    // "!level_zero:1,2" -> "level_zero:0,3,...,max"
    // * finally subtract the discard set from the accept set

    // accept  "2,*" != "*,2"
    // because "2,*" == "2,0,1,3"
    // whereas "*,2" == "0,1,2,3"
    // however
    // discard "2,*" == "*,2"

    // The std::map is sorted by its key, so this method of parsing the ODS env var
    // alters the ordering of the terms, which makes it impossible to check whether
    // all discard terms appear after all accept terms and to preserve the ordering
    // of backends as specified in the ODS string.
    // However, for single-platform requests, we are only interested in exactly one
    // backend, and we know that discard filter terms always override accept filter
    // terms, so the ordering of terms can be safely ignored -- in the special case
    // where the whole ODS string contains at most one accept term, and at most one
    // discard term, for that backend.
    // (If we wished to preserve the ordering of terms, we could replace `std::map`
    // with `std::queue<std::pair<key_type_t, value_type_t>>` or something similar.)
    auto maybeEnvVarMap = getenv_to_map("ONEAPI_DEVICE_SELECTOR", true);

    // if the ODS env var is not set at all, then pretend it was set to the default
    using EnvVarMap = std::map<std::string, std::vector<std::string>>;
    EnvVarMap mapODS = maybeEnvVarMap.has_value() ? maybeEnvVarMap.value()
                                                  : EnvVarMap{{"*", {"*"}}};

    // the full BNF grammar can be found here:
    // https://github.com/intel/llvm/blob/sycl/sycl/doc/EnvironmentVariables.md#oneapi_device_selector

    // discardFilter = "!acceptFilter"
    //  acceptFilter = "backend:filterStrings"
    // filterStrings = "filterString[,filterString[,...]]"
    //  filterString = "root[.sub[.subsub]]"
    //          root = "*|int|cpu|gpu|fpga"
    //           sub = "*|int"
    //        subsub = "*|int"

    // validation regex for filterString (not used in this code)
    std::regex validation_pattern(
        "^("
        "\\*" // C++ escape for \, regex escape for literal '*'
        "|"
        "cpu" // ensure case-insenitive, when using
        "|"
        "gpu" // ensure case-insenitive, when using
        "|"
        "fpga" // ensure case-insenitive, when using
        "|"
        "[[:digit:]]+" // '<num>'
        "|"
        "[[:digit:]]+\\.[[:digit:]]+" // '<num>.<num>'
        "|"
        "[[:digit:]]+\\.\\*" // '<num>.*.*'
        "|"
        "\\*\\.\\*" // C++ and regex escapes, literal '*.*'
        "|"
        "[[:digit:]]+\\.[[:digit:]]+\\.[[:digit:]]+" // '<num>.<num>.<num>'
        "|"
        "[[:digit:]]+\\.[[:digit:]]+\\.\\*" // '<num>.<num>.*'
        "|"
        "[[:digit:]]+\\.\\*\\.\\*" // '<num>.*.*'
        "|"
        "\\*\\.\\*\\.\\*" // C++ and regex escapes, literal '*.*.*'
        ")$",
        std::regex_constants::icase);

    ur_platform_backend_t platformBackend;
    if (UR_RESULT_SUCCESS !=
        urPlatformGetInfo(hPlatform, UR_PLATFORM_INFO_BACKEND,
                          sizeof(ur_platform_backend_t), &platformBackend, 0)) {
        return UR_RESULT_ERROR_INVALID_PLATFORM;
    }
    const std::string platformBackendName = // hPlatform->get_backend_name();
        [&platformBackend]() constexpr {
            switch (platformBackend) {
            case UR_PLATFORM_BACKEND_UNKNOWN:
                return "*"; // the only ODS string that matches
                break;
            case UR_PLATFORM_BACKEND_LEVEL_ZERO:
                return "level_zero";
                break;
            case UR_PLATFORM_BACKEND_OPENCL:
                return "opencl";
                break;
            case UR_PLATFORM_BACKEND_CUDA:
                return "cuda";
                break;
            case UR_PLATFORM_BACKEND_HIP:
                return "hip";
                break;
            case UR_PLATFORM_BACKEND_NATIVE_CPU:
                return "*"; // the only ODS string that matches
                break;
            case UR_PLATFORM_BACKEND_FORCE_UINT32:
                return ""; // no ODS string matches this
                break;
            default:
                return ""; // no ODS string matches this
                break;
            }
        }();

    using DeviceHardwareType = ur_device_type_t;

    enum class DevicePartLevel { ROOT, SUB, SUBSUB };

    using DeviceIdType = unsigned long;
    constexpr DeviceIdType DeviceIdTypeALL =
        -1; // ULONG_MAX but without #include <climits>

    struct DeviceSpec {
        DevicePartLevel level;
        DeviceHardwareType hwType = ::UR_DEVICE_TYPE_ALL;
        DeviceIdType rootId = DeviceIdTypeALL;
        DeviceIdType subId = DeviceIdTypeALL;
        DeviceIdType subsubId = DeviceIdTypeALL;
        ur_device_handle_t urDeviceHandle;
    };

    auto getRootHardwareType =
        [](const std::string &input) -> DeviceHardwareType {
        std::string lowerInput(input);
        std::transform(lowerInput.cbegin(), lowerInput.cend(),
                       lowerInput.begin(), ::tolower);
        if (lowerInput == "cpu") {
            return ::UR_DEVICE_TYPE_CPU;
        }
        if (lowerInput == "gpu") {
            return ::UR_DEVICE_TYPE_GPU;
        }
        if (lowerInput == "fpga") {
            return ::UR_DEVICE_TYPE_FPGA;
        }
        return ::UR_DEVICE_TYPE_ALL;
    };

    auto getDeviceId = [&](const std::string &input) -> DeviceIdType {
        if (input.find_first_not_of("0123456789") == std::string::npos) {
            return std::stoul(input);
        }
        return DeviceIdTypeALL;
    };

    std::vector<DeviceSpec> acceptDeviceList;
    std::vector<DeviceSpec> discardDeviceList;

    std::vector<std::string> acceptFilters;
    std::vector<std::string> discardFilters;
    for (auto &termPair : mapODS) {
        std::string backend = termPair.first;
        if (backend.empty()) {
            // malformed term: missing backend -- output ERROR, then continue
            // TODO: replace std::cout with URT message output mechanism
            std::cout << "ERROR: missing backend, format of filter = "
                         "'[!]backend:filterStrings'";
            continue;
        }
        enum FilterType {
            AcceptFilter,
            DiscardFilter,
        } termType = (backend.front() != '!') ? AcceptFilter : DiscardFilter;
        auto &deviceList = acceptDeviceList;
        if (termType != AcceptFilter) {
            backend.erase(backend.cbegin());
            deviceList = discardDeviceList;
        }
        // Note the hPlatform -> platformBackend -> platformBackendName conversion above
        // guarantees minimal sanity for the comparison with backend from the ODS string
        if (backend != "*" &&
            std::equal(platformBackendName.cbegin(), platformBackendName.cend(),
                       backend.cbegin(), backend.cend(),
                       [](const auto &a, const auto &b) {
                           // case-insensitive comparison by converting both tolower
                           return std::tolower(static_cast<unsigned char>(a)) ==
                                  std::tolower(static_cast<unsigned char>(b));
                       })) {
            // irrelevant term for current request: different backend -- silently ignore
            continue;
        }
        if (termPair.second.size() == 0) {
            // malformed term: missing filterStrings -- output ERROR, then continue
            // TODO: replace std::cout with URT message output mechanism
            std::cout << "ERROR missing filterStrings, format of filter = "
                         "'[!]backend:filterStrings'";
            continue;
        }
        if (std::find_if(termPair.second.cbegin(), termPair.second.cend(),
                         [](const auto &s) { return s.empty(); }) !=
            termPair.second.cend()) {
            // malformed term: missing filterString -- output warning, then continue
            // TODO: replace std::cout with URT message output mechanism
            std::cout << "WARNING: empty filterString, format of filterStrings "
                         "= 'filterString[,filterString[,...]]'";
            continue;
        }
        if (std::find_if(termPair.second.cbegin(), termPair.second.cend(),
                         [](const auto &s) {
                             return std::count(s.cbegin(), s.cend(), '.') > 2;
                         }) != termPair.second.cend()) {
            // malformed term: too many dots in filterString -- output warning, then continue
            // TODO: replace std::cout with URT message output mechanism
            std::cout << "WARNING: too many dots in filterString, format of "
                         "filterString = 'root[.sub[.subsub]]'";
            continue;
        }
        if (std::find_if(
                termPair.second.cbegin(), termPair.second.cend(),
                [](const auto &s) {
                    // GOOD: "*.*", "1.*.*", "*.*.*"
                    // BAD: "*.1", "*.", "1.*.2", "*.gpu"
                    std::string prefix = "*."; // every "*." pattern ...
                    std::string whole = "*.*"; // ... must be start of "*.*"
                    std::string::size_type pos = 0;
                    while ((pos = s.find(prefix, pos)) != std::string::npos) {
                        if (s.substr(pos, whole.size()) != whole) {
                            return true; // found a BAD thing, either "\*\.$" or "\*\.[^*]"
                        }
                        pos += prefix.size();
                    }
                    return false; // no BAD things, so must be okay
                }) != termPair.second.cend()) {
            // malformed term: star dot no-star in filterString -- output warning, then continue
            // TODO: replace std::cout with URT message output mechanism
            std::cout
                << "WARNING: invalid wildcard in filterString, '*.' => '*.*'";
            continue;
        }

        // TODO -- use regex validation_pattern to catch all other syntax errors in the ODS string

        for (auto &filterString : termPair.second) {
            std::string::size_type locationDot1 = filterString.find('.');
            if (locationDot1 != std::string::npos) {
                std::string firstPart = filterString.substr(0, locationDot1);
                const auto hardwareType = getRootHardwareType(firstPart);
                const auto firstDeviceId = getDeviceId(firstPart);
                // first dot found, look for another
                std::string::size_type locationDot2 =
                    filterString.find('.', locationDot1);
                std::string secondPart = filterString.substr(
                    locationDot1 + 1, locationDot2 == std::string::npos
                                          ? std::string::npos
                                          : locationDot2 - locationDot1);
                const auto secondDeviceId = getDeviceId(secondPart);
                if (locationDot2 != std::string::npos) {
                    // second dot found, this is a subsubdevice
                    std::string thirdPart =
                        filterString.substr(locationDot2 + 1);
                    const auto thirdDeviceId = getDeviceId(thirdPart);
                    deviceList.push_back(DeviceSpec{
                        DevicePartLevel::SUBSUB, hardwareType, firstDeviceId,
                        secondDeviceId, thirdDeviceId});
                } else {
                    // second dot not found, this is a subdevice
                    deviceList.push_back(DeviceSpec{DevicePartLevel::SUB,
                                                    hardwareType, firstDeviceId,
                                                    secondDeviceId});
                }
            } else {
                // first dot not found, this is a root device
                const auto hardwareType = getRootHardwareType(filterString);
                const auto firstDeviceId = getDeviceId(filterString);
                deviceList.push_back(DeviceSpec{DevicePartLevel::ROOT,
                                                hardwareType, firstDeviceId});
            }
        }

        if (termType != AcceptFilter) {
            discardFilters.insert(discardFilters.end(),
                                  termPair.second.cbegin(),
                                  termPair.second.cend());
        } else {
            acceptFilters.insert(acceptFilters.end(), termPair.second.cbegin(),
                                 termPair.second.cend());
        }
    }

    // if no accept filters are specified by the user, we must add a default "all root devices"
    if (acceptFilters.size() == 0) {
        acceptFilters.insert(acceptFilters.end(), 1, "*");
    }
    if (acceptDeviceList.size() == 0) {
        acceptDeviceList.push_back(DeviceSpec{
            DevicePartLevel::ROOT, ::UR_DEVICE_TYPE_ALL, DeviceIdTypeALL});
    }

    std::vector<DeviceSpec> rootDevices;
    std::vector<DeviceSpec> subDevices;
    std::vector<DeviceSpec> subSubDevices;

    // To support root device terms:
    {
        uint32_t platformNumRootDevicesAll = 0;
        if (UR_RESULT_SUCCESS != urDeviceGet(hPlatform, UR_DEVICE_TYPE_ALL, 0,
                                             nullptr,
                                             &platformNumRootDevicesAll)) {
            return UR_RESULT_ERROR_DEVICE_NOT_FOUND;
        }
        std::vector<ur_device_handle_t> rootDeviceHandles(
            platformNumRootDevicesAll);
        auto pRootDevices = rootDeviceHandles.data();
        if (UR_RESULT_SUCCESS != urDeviceGet(hPlatform, UR_DEVICE_TYPE_ALL,
                                             platformNumRootDevicesAll,
                                             pRootDevices, 0)) {
            return UR_RESULT_ERROR_DEVICE_NOT_FOUND;
        }

        DeviceIdType deviceCount = 0;
        std::transform(
            rootDeviceHandles.cbegin(), rootDeviceHandles.cend(),
            std::back_inserter(rootDevices),
            [&](ur_device_handle_t urDeviceHandle) {
                // obtain and record device type from platform (squash errors)
                ur_device_type_t hardwareType = ::UR_DEVICE_TYPE_DEFAULT;
                urDeviceGetInfo(urDeviceHandle, UR_DEVICE_INFO_TYPE,
                                sizeof(ur_device_type_t), &hardwareType, 0);
                return DeviceSpec{DevicePartLevel::ROOT, hardwareType,
                                  deviceCount++,         DeviceIdTypeALL,
                                  DeviceIdTypeALL,       urDeviceHandle};
            });

        // apply the function parameter: ur_device_type_t DeviceType
        // remove_if(..., urDeviceHandle->deviceType == DeviceType)
        rootDevices.erase(
            std::remove_if(
                rootDevices.begin(), rootDevices.end(),
                [DeviceType](DeviceSpec &device) {
                    const bool keep =
                        (DeviceType ==
                         DeviceHardwareType::UR_DEVICE_TYPE_ALL) ||
                        (DeviceType ==
                         DeviceHardwareType::UR_DEVICE_TYPE_DEFAULT) ||
                        (DeviceType == device.hwType);
                    return !keep;
                }),
            rootDevices.end());
    }

    // To support sub-device terms:
    std::for_each(
        rootDevices.cbegin(), rootDevices.cend(), [&](DeviceSpec device) {
            ur_device_partition_property_t propNextPart{
                UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                {UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE}};
            ur_device_partition_properties_t partitionProperties{
                UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES, nullptr,
                &propNextPart, 1};
            uint32_t numSubdevices = 0;
            if (UR_RESULT_SUCCESS !=
                urDevicePartition(device.urDeviceHandle, &partitionProperties,
                                  0, nullptr, &numSubdevices)) {
                return UR_RESULT_ERROR_DEVICE_PARTITION_FAILED;
            }
            std::vector<ur_device_handle_t> subDeviceHandles(numSubdevices);
            auto pSubDevices = subDeviceHandles.data();
            if (UR_RESULT_SUCCESS !=
                urDevicePartition(device.urDeviceHandle, &partitionProperties,
                                  numSubdevices, pSubDevices, 0)) {
                return UR_RESULT_ERROR_DEVICE_PARTITION_FAILED;
            }
            DeviceIdType subDeviceCount = 0;
            std::transform(subDeviceHandles.cbegin(), subDeviceHandles.cend(),
                           std::back_inserter(subDevices),
                           [&](ur_device_handle_t urDeviceHandle) {
                               return DeviceSpec{
                                   DevicePartLevel::SUB, device.hwType,
                                   device.rootId,        subDeviceCount++,
                                   DeviceIdTypeALL,      urDeviceHandle};
                           });
            return UR_RESULT_SUCCESS;
        });

    // To support sub-sub-device terms:
    std::for_each(
        subDevices.cbegin(), subDevices.cend(), [&](DeviceSpec device) {
            ur_device_partition_property_t propNextPart{
                UR_DEVICE_PARTITION_BY_AFFINITY_DOMAIN,
                {UR_DEVICE_AFFINITY_DOMAIN_FLAG_NEXT_PARTITIONABLE}};
            ur_device_partition_properties_t partitionProperties{
                UR_STRUCTURE_TYPE_DEVICE_PARTITION_PROPERTIES, nullptr,
                &propNextPart, 1};
            uint32_t numSubSubdevices = 0;
            if (UR_RESULT_SUCCESS !=
                urDevicePartition(device.urDeviceHandle, &partitionProperties,
                                  0, nullptr, &numSubSubdevices)) {
                return UR_RESULT_ERROR_DEVICE_PARTITION_FAILED;
            }
            std::vector<ur_device_handle_t> subSubDeviceHandles(
                numSubSubdevices);
            auto pSubSubDevices = subSubDeviceHandles.data();
            if (UR_RESULT_SUCCESS !=
                urDevicePartition(device.urDeviceHandle, &partitionProperties,
                                  numSubSubdevices, pSubSubDevices, 0)) {
                return UR_RESULT_ERROR_DEVICE_PARTITION_FAILED;
            }
            DeviceIdType subSubDeviceCount = 0;
            std::transform(
                subSubDeviceHandles.cbegin(), subSubDeviceHandles.cend(),
                std::back_inserter(subSubDevices),
                [&](ur_device_handle_t urDeviceHandle) {
                    return DeviceSpec{DevicePartLevel::SUBSUB, device.hwType,
                                      device.rootId,           device.subId,
                                      subSubDeviceCount++,     urDeviceHandle};
                });
            return UR_RESULT_SUCCESS;
        });

    auto ApplyFilter = [&](DeviceSpec &filter, DeviceSpec &device) -> bool {
        bool matches = false;
        if (filter.rootId == DeviceIdTypeALL) {
            // if this is a root device filter, then it must be '*' or 'cpu' or 'gpu' or 'fpga'
            // if this is a subdevice filter, then it must be '*.*'
            // if this is a subsubdevice filter, then it must be '*.*.*'
            matches = (filter.hwType == device.hwType) ||
                      (filter.hwType == DeviceHardwareType::UR_DEVICE_TYPE_ALL);
        } else if (filter.rootId != device.rootId) {
            // root part in filter is a number but does not match the number in the root part of device
            matches = false;
        } else if (filter.level == DevicePartLevel::ROOT) {
            // this is a root device filter with a number that matches
            matches = true;
        } else if (filter.subId == DeviceIdTypeALL) {
            // sub type of star always matches (when root part matches, which we already know here)
            // if this is a subdevice filter, then it must be 'matches.*'
            // if this is a subsubdevice filter, then it must be 'matches.*.*'
            matches = true;
        } else if (filter.subId != device.subId) {
            // sub part in filter is a number but does not match the number in the sub part of device
            matches = false;
        } else if (filter.level == DevicePartLevel::SUB) {
            // this is a sub device number filter, numbers match in both parts
            matches = true;
        } else if (filter.subsubId == DeviceIdTypeALL) {
            // subsub type of star always matches (when other parts match, which we already know here)
            // this is a subsub device filter, it must be 'matches.matches.*'
            matches = true;
        } else {
            // this is a subsub device filter, numbers in all three parts match
            matches = (filter.subsubId == device.subsubId);
        }
        return matches;
    };

    // apply each discard filter in turn by removing all matching elements
    // from the appropriate device handle vector returned by the platform;
    // no side-effect: the matching devices are just removed and discarded
    for (auto &discard : discardDeviceList) {
        auto ApplyDiscardFilter = [&](auto &device) -> bool {
            return ApplyFilter(discard, device);
        };
        if (discard.level == DevicePartLevel::ROOT) {
            rootDevices.erase(std::remove_if(rootDevices.begin(),
                                             rootDevices.end(),
                                             ApplyDiscardFilter),
                              rootDevices.end());
        }
        if (discard.level == DevicePartLevel::SUB) {
            subDevices.erase(std::remove_if(subDevices.begin(),
                                            subDevices.end(),
                                            ApplyDiscardFilter),
                             subDevices.end());
        }
        if (discard.level == DevicePartLevel::SUBSUB) {
            subSubDevices.erase(std::remove_if(subSubDevices.begin(),
                                               subSubDevices.end(),
                                               ApplyDiscardFilter),
                                subSubDevices.end());
        }
    }

    std::vector<ur_device_handle_t> selectedDevices;

    // apply each accept filter in turn by removing all matching elements
    // from the appropriate device handle vector returned by the platform
    // but using a predicate with a side-effect that takes a copy of each
    // of the accepted device handles just before they are removed
    // removing each item as it is selected prevents us taking duplicates
    // without needing O(n^2) de-duplicatation or symbolic simplification
    for (auto &accept : acceptDeviceList) {
        auto ApplyAcceptFilter = [&](auto &device) -> bool {
            const bool matches = ApplyFilter(accept, device);
            if (matches) {
                selectedDevices.push_back(device.urDeviceHandle);
            }
            return matches;
        };
        if (accept.level == DevicePartLevel::ROOT) {
            rootDevices.erase(std::remove_if(rootDevices.begin(),
                                             rootDevices.end(),
                                             ApplyAcceptFilter),
                              rootDevices.end());
        }
        if (accept.level == DevicePartLevel::SUB) {
            subDevices.erase(std::remove_if(subDevices.begin(),
                                            subDevices.end(),
                                            ApplyAcceptFilter),
                             subDevices.end());
        }
        if (accept.level == DevicePartLevel::SUBSUB) {
            subSubDevices.erase(std::remove_if(subSubDevices.begin(),
                                               subSubDevices.end(),
                                               ApplyAcceptFilter),
                                subSubDevices.end());
        }
    }

    // selectedDevices is now a vector containing all the right device handles

    // should we return the size of the vector or the content of the vector?
    if (NumEntries == 0) {
        *pNumDevices = static_cast<uint32_t>(selectedDevices.size());
    } else if (NumEntries > 0) {
        size_t numToCopy = std::min((size_t)NumEntries, selectedDevices.size());
        std::copy_n(selectedDevices.cbegin(), numToCopy, phDevices);
        if (pNumDevices != nullptr) {
            *pNumDevices = static_cast<uint32_t>(numToCopy);
            return UR_RESULT_ERROR_ADAPTER_SPECIFIC;
        }
    }

    return UR_RESULT_SUCCESS;
}
} // namespace ur_lib
