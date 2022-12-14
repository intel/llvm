//===--- usm_allocator_config.cpp -configuration for USM memory allocator---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "usm_allocator_config.hpp"

#include <iomanip>
#include <iostream>
#include <string>

namespace usm_settings {

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

USMAllocatorConfig::USMAllocatorConfig() {
  size_t i = 0;
  for (auto &memoryTypeName : MemTypeNames) {
    Configs[i++].memoryTypeName = memoryTypeName;
  }

  // Buckets for Host use a minimum of the cache line size of 64 bytes.
  // This prevents two separate allocations residing in the same cache line.
  // Buckets for Device and Shared allocations will use starting size of 512.
  // This is because memory compression on newer GPUs makes the
  // minimum granularity 512 bytes instead of 64.
  Configs[MemType::Host].MinBucketSize = 64;
  Configs[MemType::Device].MinBucketSize = 512;
  Configs[MemType::Shared].MinBucketSize = 512;
  Configs[MemType::SharedReadOnly].MinBucketSize = 512;

  // Initialize default pool settings.
  Configs[MemType::Host].MaxPoolableSize = 2_MB;
  Configs[MemType::Host].Capacity = 4;
  Configs[MemType::Host].SlabMinSize = 64_KB;

  Configs[MemType::Device].MaxPoolableSize = 4_MB;
  Configs[MemType::Device].Capacity = 4;
  Configs[MemType::Device].SlabMinSize = 64_KB;

  // Disable pooling of shared USM allocations.
  Configs[MemType::Shared].MaxPoolableSize = 0;
  Configs[MemType::Shared].Capacity = 0;
  Configs[MemType::Shared].SlabMinSize = 2_MB;

  // Allow pooling of shared allocations that are only modified on host.
  Configs[MemType::SharedReadOnly].MaxPoolableSize = 4_MB;
  Configs[MemType::SharedReadOnly].Capacity = 4;
  Configs[MemType::SharedReadOnly].SlabMinSize = 2_MB;

  // Parse optional parameters of this form:
  // SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=[EnableBuffers][;[MaxPoolSize][;memtypelimits]...]
  //  memtypelimits: [<memtype>:]<limits>
  //  memtype: host|device|shared
  //  limits:  [MaxPoolableSize][,[Capacity][,SlabMinSize]]
  //
  // Without a memory type, the limits are applied to each memory type.
  // Parameters are for each context, except MaxPoolSize, which is overall
  // pool size for all contexts.
  // Duplicate specifications will result in the right-most taking effect.
  //
  // EnableBuffers:   Apply chunking/pooling to SYCL buffers.
  //                  Default 1.
  // MaxPoolSize:     Limit on overall unfreed memory.
  //                  Default 16MB.
  // MaxPoolableSize: Maximum allocation size subject to chunking/pooling.
  //                  Default 2MB host, 4MB device and 0 shared.
  // Capacity:        Maximum number of unfreed allocations in each bucket.
  //                  Default 4.
  // SlabMinSize:     Minimum allocation size requested from USM.
  //                  Default 64KB host and device, 2MB shared.
  //
  // Example of usage:
  // SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR=1;32M;host:1M,4,64K;device:1M,4,64K;shared:0,0,2M

  auto GetValue = [=](std::string &Param, size_t Length, size_t &Setting) {
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
    if (TheNumber.find_first_not_of("0123456789") == std::string::npos)
      Setting = std::stoi(TheNumber) * Multiplier;
  };

  auto ParamParser = [=](std::string &Params, size_t &Setting,
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

  auto MemParser = [=](std::string &Params, MemType M) {
    bool ParamWasSet;
    MemType LM = M;
    if (M == MemType::All)
      LM = MemType::Host;

    bool More = ParamParser(Params, Configs[LM].MaxPoolableSize, ParamWasSet);
    if (ParamWasSet && M == MemType::All) {
      for (auto &Config : Configs) {
        Config.MaxPoolableSize = Configs[LM].MaxPoolableSize;
      }
    }
    if (More) {
      More = ParamParser(Params, Configs[LM].Capacity, ParamWasSet);
      if (ParamWasSet && M == MemType::All) {
        for (auto &Config : Configs) {
          Config.Capacity = Configs[LM].Capacity;
        }
      }
    }
    if (More) {
      ParamParser(Params, Configs[LM].SlabMinSize, ParamWasSet);
      if (ParamWasSet && M == MemType::All) {
        for (auto &Config : Configs) {
          Config.SlabMinSize = Configs[LM].SlabMinSize;
        }
      }
    }
  };

  auto MemTypeParser = [=](std::string &Params) {
    int Pos = 0;
    MemType M = MemType::All;
    if (Params.compare(0, 5, "host:") == 0) {
      Pos = 5;
      M = MemType::Host;
    } else if (Params.compare(0, 7, "device:") == 0) {
      Pos = 7;
      M = MemType::Device;
    } else if (Params.compare(0, 7, "shared:") == 0) {
      Pos = 7;
      M = MemType::Shared;
    } else if (Params.compare(0, 17, "read_only_shared:") == 0) {
      Pos = 17;
      M = MemType::SharedReadOnly;
    }
    if (Pos > 0)
      Params.erase(0, Pos);
    MemParser(Params, M);
  };

  auto limits = std::make_shared<USMLimits>();

  // Update pool settings if specified in environment.
  char *PoolParams = getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR");
  if (PoolParams != nullptr) {
    std::string Params(PoolParams);
    size_t Pos = Params.find(';');
    if (Pos != std::string::npos) {
      if (Pos > 0) {
        GetValue(Params, Pos, EnableBuffers);
      }
      Params.erase(0, Pos + 1);
      size_t Pos = Params.find(';');
      if (Pos != std::string::npos) {
        if (Pos > 0) {
          GetValue(Params, Pos, limits->MaxSize);
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
            if (Params.size() == 0)
              break;
          } else {
            MemTypeParser(Params);
            break;
          }
        } while (true);
      } else {
        // set MaxPoolSize for all configs
        GetValue(Params, Params.size(), limits->MaxSize);
      }
    } else {
      GetValue(Params, Params.size(), EnableBuffers);
    }
  }

  char *PoolTraceVal = getenv("SYCL_PI_LEVEL_ZERO_USM_ALLOCATOR_TRACE");
  int PoolTrace = 0;
  if (PoolTraceVal != nullptr) {
    PoolTrace = std::atoi(PoolTraceVal);
  }

  for (auto &Config : Configs) {
    Config.limits = limits;
    Config.PoolTrace = PoolTrace;
  }

  if (PoolTrace < 1)
    return;

  std::cout << "USM Pool Settings (Built-in or Adjusted by Environment "
               "Variable)"
            << std::endl;

  std::cout << std::setw(15) << "Parameter" << std::setw(12) << "Host"
            << std::setw(12) << "Device" << std::setw(12) << "Shared RW"
            << std::setw(12) << "Shared RO" << std::endl;
  std::cout << std::setw(15) << "SlabMinSize" << std::setw(12)
            << Configs[MemType::Host].SlabMinSize << std::setw(12)
            << Configs[MemType::Device].SlabMinSize << std::setw(12)
            << Configs[MemType::Shared].SlabMinSize << std::setw(12)
            << Configs[MemType::SharedReadOnly].SlabMinSize << std::endl;
  std::cout << std::setw(15) << "MaxPoolableSize" << std::setw(12)
            << Configs[MemType::Host].MaxPoolableSize << std::setw(12)
            << Configs[MemType::Device].MaxPoolableSize << std::setw(12)
            << Configs[MemType::Shared].MaxPoolableSize << std::setw(12)
            << Configs[MemType::SharedReadOnly].MaxPoolableSize << std::endl;
  std::cout << std::setw(15) << "Capacity" << std::setw(12)
            << Configs[MemType::Host].Capacity << std::setw(12)
            << Configs[MemType::Device].Capacity << std::setw(12)
            << Configs[MemType::Shared].Capacity << std::setw(12)
            << Configs[MemType::SharedReadOnly].Capacity << std::endl;
  std::cout << std::setw(15) << "MaxPoolSize" << std::setw(12)
            << limits->MaxSize << std::endl;
  std::cout << std::setw(15) << "EnableBuffers" << std::setw(12)
            << EnableBuffers << std::endl
            << std::endl;
}
} // namespace usm_settings
