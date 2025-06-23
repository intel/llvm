/*
 *
 * Copyright (C) 2022-2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include <ur_api.h>

#include <cstdint>
#include <optional>
#include <string_view>

class HipOffloadBundleParser {
  static constexpr std::string_view Magic = "__CLANG_OFFLOAD_BUNDLE__";
  const uint8_t *Buff;
  size_t Length;

  struct __attribute__((packed)) BundleEntry {
    uint64_t ObjectOffset;
    uint64_t ObjectSize;
    uint64_t EntryIdSize;
    char EntryIdStart;
  };

  struct __attribute__((packed)) BundleHeader {
    const char HeaderMagic[Magic.size()];
    uint64_t EntryCount;
    BundleEntry FirstEntry;
  };

  HipOffloadBundleParser() = delete;
  HipOffloadBundleParser(const uint8_t *Buff, size_t Length)
      : Buff(Buff), Length(Length) {}

public:
  static std::optional<HipOffloadBundleParser> load(const uint8_t *Buff,
                                                    size_t Length);

  ur_result_t extract(std::string_view SearchTargetId,
                      const uint8_t *&OutBinary, size_t &OutLength);

  std::optional<BundleEntry> containsBundle(std::string_view SearchTargetId);
};
