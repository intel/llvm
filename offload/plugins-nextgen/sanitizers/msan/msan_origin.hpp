//===--- msan_origin.hpp - MSan origin ids --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include "sanitizer_stackdepot.hpp"
#include "sanitizer_stacktrace.hpp"

#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>

namespace llvm::offload::sanitizer {
namespace msan {

// Origin handling.
//
//   10xx xxxx xxxx xxxx   device USM
//   110x xxxx xxxx xxxx   host USM
//   1110 xxxx xxxx xxxx   shared USM
//   1111 0xxx xxxx xxxx   local memory
//   1111 1xxx xxxx xxxx   exportable memory
//   0000 xxxx xxxx xxxx   private memory
//   0zzz xxxx xxxx xxxx   chained
//

class Origin {
public:
  const uint32_t *rawId() const { return &raw_id_; }

  bool isHeapOrigin() const {
    return isDeviceUSMOrigin() || isHostUSMOrigin() || isSharedUSMOrigin() ||
           isLocalOrigin() || isExportableMemOrigin();
  }

  HeapType getHeapType() const {
    if (isDeviceUSMOrigin())
      return HeapType::DeviceUSM;
    if (isHostUSMOrigin())
      return HeapType::HostUSM;
    if (isSharedUSMOrigin())
      return HeapType::SharedUSM;
    if (isLocalOrigin())
      return HeapType::Local;
    if (isExportableMemOrigin())
      return HeapType::ExportableMem;

    assert(false && "Unknown heap type");
    return HeapType::DeviceUSM; // Default fallback, should never reach here
  }

  bool isDeviceUSMOrigin() const {
    // 10xx xxxx xxxx xxxx
    return raw_id_ >> kDeviceUSMShift == kDeviceUSMBits;
  }
  bool isHostUSMOrigin() const {
    // 110x xxxx xxxx xxxx
    return raw_id_ >> kHostUSMShift == kHostUSMBits;
  }
  bool isSharedUSMOrigin() const {
    // 1110 xxxx xxxx xxxx
    return raw_id_ >> kSharedUSMShift == kSharedUSMBits;
  }

  bool isLocalOrigin() const {
    // 1111 0xxx xxxx xxxx
    return raw_id_ >> kLocalShift == kLocalBits;
  }

  bool isExportableMemOrigin() const {
    // 1111 1xxx xxxx xxxx
    return raw_id_ >> kExportableMemShift == kExportableMemBits;
  }

  StackTrace getHeapStackTrace() const {
    assert(isHeapOrigin());
    return StackDepotGet(raw_id_);
  }

  static Origin CreateHeapOrigin(StackTrace &Stack, HeapType Type) {
    static std::array<std::atomic_uint32_t, kHeapTypeCount> _NextIds;
    uint32_t StackId = _NextIds[(uint32_t)Type].fetch_add(1);

    switch (Type) {
    case HeapType::DeviceUSM:
      assert((StackId & kDeviceUSMIdMask) == StackId);
      StackId = (kDeviceUSMBits << kDeviceUSMShift) | StackId;
      break;
    case HeapType::HostUSM:
      assert((StackId & kHostUSMIdMask) == StackId);
      StackId = (kHostUSMBits << kHostUSMShift) | StackId;
      break;
    case HeapType::SharedUSM:
      assert((StackId & kSharedUSMIdMask) == StackId);
      StackId = (kSharedUSMBits << kSharedUSMShift) | StackId;
      break;
    case HeapType::Local:
      assert((StackId & kLocalIdMask) == StackId);
      StackId = (kLocalBits << kLocalShift) | StackId;
      break;
    case HeapType::ExportableMem:
      assert((StackId & kExportableMemIdMask) == StackId);
      StackId = (kExportableMemBits << kExportableMemShift) | StackId;
      break;
    default:
      assert(false && "Unknown heap type");
      StackId = 0;
    }

    StackDepotPut(StackId, Stack);

    return Origin(StackId);
  }

  static Origin FromRawId(uint32_t id) { return Origin(id); }

private:
  static const int kDeviceUSMBits = 2;
  static const int kDeviceUSMShift = 32 - 2;

  static const int kHostUSMBits = 6;
  static const int kHostUSMShift = 32 - 3;

  static const int kSharedUSMBits = 14;
  static const int kSharedUSMShift = 32 - 4;

  static const int kLocalBits = 30;
  static const int kLocalShift = 32 - 5;

  static const int kExportableMemBits = 31;
  static const int kExportableMemShift = 32 - 5;

  static const uint32_t kDeviceUSMIdMask = ((uint32_t)-1) >>
                                           (32 - kDeviceUSMShift);
  static const uint32_t kHostUSMIdMask = ((uint32_t)-1) >> (32 - kHostUSMShift);
  static const uint32_t kSharedUSMIdMask = ((uint32_t)-1) >>
                                           (32 - kSharedUSMShift);
  static const uint32_t kLocalIdMask = ((uint32_t)-1) >> (32 - kLocalShift);
  static const uint32_t kExportableMemIdMask = ((uint32_t)-1) >>
                                               (32 - kExportableMemShift);

  uint32_t raw_id_;

  explicit Origin(uint32_t raw_id) : raw_id_(raw_id) {}
};

} // namespace msan
} // namespace llvm::offload::sanitizer
