//===--- disjoint_pool_config_parser.cpp -configuration for USM memory pool-==//
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "disjoint_pool_config_parser.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <string>

namespace usm {
constexpr auto operator""_B(unsigned long long x) -> size_t { return x; }
constexpr auto operator""_KB(unsigned long long x) -> size_t {
    return x * 1024;
}
constexpr auto operator""_MB(unsigned long long x) -> size_t {
    return x * 1024 * 1024;
}
constexpr auto operator""_GB(unsigned long long x) -> size_t {
    return x * 1024 * 1024 * 1024;
}

DisjointPoolAllConfigs::DisjointPoolAllConfigs(int trace) {
    for (auto &Config : Configs) {
        Config = umfDisjointPoolParamsDefault();
        Config.PoolTrace = trace;
    }

    Configs[DisjointPoolMemType::Host].Name = "Host";
    Configs[DisjointPoolMemType::Device].Name = "Device";
    Configs[DisjointPoolMemType::Shared].Name = "Shared";
    Configs[DisjointPoolMemType::SharedReadOnly].Name = "SharedReadOnly";

    // Buckets for Host use a minimum of the cache line size of 64 bytes.
    // This prevents two separate allocations residing in the same cache line.
    // Buckets for Device and Shared allocations will use starting size of 512.
    // This is because memory compression on newer GPUs makes the
    // minimum granularity 512 bytes instead of 64.
    Configs[DisjointPoolMemType::Host].MinBucketSize = 64;
    Configs[DisjointPoolMemType::Device].MinBucketSize = 512;
    Configs[DisjointPoolMemType::Shared].MinBucketSize = 512;
    Configs[DisjointPoolMemType::SharedReadOnly].MinBucketSize = 512;

    // Initialize default pool settings.
    Configs[DisjointPoolMemType::Host].MaxPoolableSize = 2_MB;
    Configs[DisjointPoolMemType::Host].Capacity = 4;
    Configs[DisjointPoolMemType::Host].SlabMinSize = 64_KB;

    Configs[DisjointPoolMemType::Device].MaxPoolableSize = 4_MB;
    Configs[DisjointPoolMemType::Device].Capacity = 4;
    Configs[DisjointPoolMemType::Device].SlabMinSize = 64_KB;

    // Disable pooling of shared USM allocations.
    Configs[DisjointPoolMemType::Shared].MaxPoolableSize = 0;
    Configs[DisjointPoolMemType::Shared].Capacity = 0;
    Configs[DisjointPoolMemType::Shared].SlabMinSize = 2_MB;

    // Allow pooling of shared allocations that are only modified on host.
    Configs[DisjointPoolMemType::SharedReadOnly].MaxPoolableSize = 4_MB;
    Configs[DisjointPoolMemType::SharedReadOnly].Capacity = 4;
    Configs[DisjointPoolMemType::SharedReadOnly].SlabMinSize = 2_MB;
}

DisjointPoolAllConfigs parseDisjointPoolConfig(const std::string &config,
                                               int trace) {
    DisjointPoolAllConfigs AllConfigs;

    // TODO: replace with UR ENV var parser and avoid creating a copy of 'config'
    auto GetValue = [](std::string &Param, size_t Length, size_t &Setting) {
        size_t Multiplier = 1;
        if (tolower(Param[Length - 1]) == 'k') {
            Length--;
            Multiplier = 1_KB;
        }
        if (tolower(Param[Length - 1]) == 'm') {
            Length--;
            Multiplier = 1_MB;
        }
        if (tolower(Param[Length - 1]) == 'g') {
            Length--;
            Multiplier = 1_GB;
        }
        std::string TheNumber = Param.substr(0, Length);
        if (TheNumber.find_first_not_of("0123456789") == std::string::npos) {
            Setting = std::stoi(TheNumber) * Multiplier;
        }
    };

    auto ParamParser = [GetValue](std::string &Params, size_t &Setting,
                                  bool &ParamWasSet) {
        bool More;
        if (Params.size() == 0) {
            ParamWasSet = false;
            return false;
        }
        size_t Pos = Params.find(',');
        if (Pos != std::string::npos) {
            if (Pos > 0) {
                GetValue(Params, Pos, Setting);
                ParamWasSet = true;
            }
            Params.erase(0, Pos + 1);
            More = true;
        } else {
            GetValue(Params, Params.size(), Setting);
            ParamWasSet = true;
            More = false;
        }
        return More;
    };

    auto MemParser = [&AllConfigs, ParamParser](std::string &Params,
                                                DisjointPoolMemType memType =
                                                    DisjointPoolMemType::All) {
        bool ParamWasSet;
        DisjointPoolMemType LM = memType;
        if (memType == DisjointPoolMemType::All) {
            LM = DisjointPoolMemType::Host;
        }

        bool More = ParamParser(Params, AllConfigs.Configs[LM].MaxPoolableSize,
                                ParamWasSet);
        if (ParamWasSet && memType == DisjointPoolMemType::All) {
            for (auto &Config : AllConfigs.Configs) {
                Config.MaxPoolableSize = AllConfigs.Configs[LM].MaxPoolableSize;
            }
        }
        if (More) {
            More = ParamParser(Params, AllConfigs.Configs[LM].Capacity,
                               ParamWasSet);
            if (ParamWasSet && memType == DisjointPoolMemType::All) {
                for (auto &Config : AllConfigs.Configs) {
                    Config.Capacity = AllConfigs.Configs[LM].Capacity;
                }
            }
        }
        if (More) {
            ParamParser(Params, AllConfigs.Configs[LM].SlabMinSize,
                        ParamWasSet);
            if (ParamWasSet && memType == DisjointPoolMemType::All) {
                for (auto &Config : AllConfigs.Configs) {
                    Config.SlabMinSize = AllConfigs.Configs[LM].SlabMinSize;
                }
            }
        }
    };

    auto MemTypeParser = [MemParser](std::string &Params) {
        int Pos = 0;
        DisjointPoolMemType M(DisjointPoolMemType::All);
        if (Params.compare(0, 5, "host:") == 0) {
            Pos = 5;
            M = DisjointPoolMemType::Host;
        } else if (Params.compare(0, 7, "device:") == 0) {
            Pos = 7;
            M = DisjointPoolMemType::Device;
        } else if (Params.compare(0, 7, "shared:") == 0) {
            Pos = 7;
            M = DisjointPoolMemType::Shared;
        } else if (Params.compare(0, 17, "read_only_shared:") == 0) {
            Pos = 17;
            M = DisjointPoolMemType::SharedReadOnly;
        }
        if (Pos > 0) {
            Params.erase(0, Pos);
        }
        MemParser(Params, M);
    };

    size_t MaxSize = (std::numeric_limits<size_t>::max)();

    // Update pool settings if specified in environment.
    size_t EnableBuffers = 1;
    if (config != "") {
        std::string Params = config;
        size_t Pos = Params.find(';');
        if (Pos != std::string::npos) {
            if (Pos > 0) {
                GetValue(Params, Pos, EnableBuffers);
            }
            Params.erase(0, Pos + 1);
            size_t Pos = Params.find(';');
            if (Pos != std::string::npos) {
                if (Pos > 0) {
                    GetValue(Params, Pos, MaxSize);
                }
                Params.erase(0, Pos + 1);
                do {
                    size_t Pos = Params.find(';');
                    if (Pos != std::string::npos) {
                        if (Pos > 0) {
                            std::string MemParams = Params.substr(0, Pos);
                            MemTypeParser(MemParams);
                        }
                        Params.erase(0, Pos + 1);
                        if (Params.size() == 0) {
                            break;
                        }
                    } else {
                        MemTypeParser(Params);
                        break;
                    }
                } while (true);
            } else {
                // set MaxPoolSize for all configs
                GetValue(Params, Params.size(), MaxSize);
            }
        } else {
            GetValue(Params, Params.size(), EnableBuffers);
        }
    }

    AllConfigs.limits = std::shared_ptr<umf_disjoint_pool_shared_limits_t>(
        umfDisjointPoolSharedLimitsCreate(MaxSize),
        umfDisjointPoolSharedLimitsDestroy);

    for (auto &Config : AllConfigs.Configs) {
        Config.SharedLimits = AllConfigs.limits.get();
        Config.PoolTrace = trace;
    }

    if (!EnableBuffers) {
        return {};
    }

    if (!trace) {
        return AllConfigs;
    }

    std::cout << "USM Pool Settings (Built-in or Adjusted by Environment "
                 "Variable)"
              << std::endl;

    std::cout << std::setw(15) << "Parameter" << std::setw(12) << "Host"
              << std::setw(12) << "Device" << std::setw(12) << "Shared RW"
              << std::setw(12) << "Shared RO" << std::endl;
    std::cout
        << std::setw(15) << "SlabMinSize" << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::Host].SlabMinSize
        << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::Device].SlabMinSize
        << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::Shared].SlabMinSize
        << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::SharedReadOnly].SlabMinSize
        << std::endl;
    std::cout << std::setw(15) << "MaxPoolableSize" << std::setw(12)
              << AllConfigs.Configs[DisjointPoolMemType::Host].MaxPoolableSize
              << std::setw(12)
              << AllConfigs.Configs[DisjointPoolMemType::Device].MaxPoolableSize
              << std::setw(12)
              << AllConfigs.Configs[DisjointPoolMemType::Shared].MaxPoolableSize
              << std::setw(12)
              << AllConfigs.Configs[DisjointPoolMemType::SharedReadOnly]
                     .MaxPoolableSize
              << std::endl;
    std::cout
        << std::setw(15) << "Capacity" << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::Host].Capacity
        << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::Device].Capacity
        << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::Shared].Capacity
        << std::setw(12)
        << AllConfigs.Configs[DisjointPoolMemType::SharedReadOnly].Capacity
        << std::endl;
    std::cout << std::setw(15) << "MaxPoolSize" << std::setw(12) << MaxSize
              << std::endl;
    std::cout << std::setw(15) << "EnableBuffers" << std::setw(12)
              << EnableBuffers << std::endl
              << std::endl;

    return AllConfigs;
}
} // namespace usm
