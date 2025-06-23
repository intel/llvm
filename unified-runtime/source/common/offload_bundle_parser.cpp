/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 */

#include "offload_bundle_parser.hpp"
#include <string>

std::optional<HipOffloadBundleParser>
HipOffloadBundleParser::load(const uint8_t *Buff, size_t Length) {
  if (std::string_view{reinterpret_cast<const char *>(Buff), Length}.find(
          Magic) != 0) {
    return std::nullopt;
  }
  return HipOffloadBundleParser(Buff, Length);
}

ur_result_t HipOffloadBundleParser::extract(std::string_view SearchTargetId,
                                            const uint8_t *&OutBinary,
                                            size_t &OutLength) {
  const char *Limit = reinterpret_cast<const char *>(&Buff[Length]);
  auto Entry = containsBundle(SearchTargetId);
  if (!Entry) {
    return UR_RESULT_ERROR_INVALID_PROGRAM;
  }
  OutBinary = reinterpret_cast<const uint8_t *>(&Buff[Entry->ObjectOffset]);
  OutLength = Entry->ObjectSize;

  if (reinterpret_cast<const char *>(&OutBinary[OutLength]) > Limit) {
    return UR_RESULT_ERROR_INVALID_PROGRAM;
  }
  return UR_RESULT_SUCCESS;
}

std::optional<HipOffloadBundleParser::BundleEntry>
HipOffloadBundleParser::containsBundle(std::string_view SearchTargetId) {
  const char *Limit = reinterpret_cast<const char *>(&Buff[Length]);

  // The different check here means that a binary consisting of only the magic
  // bytes (but nothing else) will result in INVALID_PROGRAM rather than being
  // treated as a non-bundle
  auto *Header = reinterpret_cast<const BundleHeader *>(Buff);
  if (reinterpret_cast<const char *>(&Header->FirstEntry) > Limit) {
    return std::nullopt;
  }

  // std::string_view::ends_with is C++20. Until then, roll our own. Note then
  // this is the equivalent form listed on en.cppreference.com.
  auto ends_with = [](std::string_view str, std::string_view sv) {
    return str.size() >= sv.size() &&
           str.compare(str.size() - sv.size(), std::string::npos, sv) == 0;
  };

  const auto *CurrentEntry = &Header->FirstEntry;
  for (uint64_t I = 0; I < Header->EntryCount; I++) {
    if (&CurrentEntry->EntryIdStart > Limit) {
      return std::nullopt;
    }
    auto EntryId = std::string_view(&CurrentEntry->EntryIdStart,
                                    CurrentEntry->EntryIdSize);
    if (EntryId.end() > Limit) {
      return std::nullopt;
    }

    // Will match either "hip" or "hipv4"
    bool isHip = EntryId.find("hip") == 0;
    bool VersionMatches = ends_with(EntryId, SearchTargetId);

    if (isHip && VersionMatches)
      return *CurrentEntry;

    CurrentEntry = reinterpret_cast<const BundleEntry *>(EntryId.end());
  }

  return std::nullopt;
}
