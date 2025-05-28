/*
 *
 * Copyright (C) 2025 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file msan_origin.hpp
 *
 */

#pragma once

#include "sanitizer_common/sanitizer_stackdepot.hpp"
#include "sanitizer_common/sanitizer_stacktrace.hpp"

#include <cassert>
#include <cstdint>

namespace ur_sanitizer_layer {
namespace msan {

// Origin handling.
//
//   10xx xxxx xxxx xxxx   device USM
//   110x xxxx xxxx xxxx   host USM
//   1110 xxxx xxxx xxxx   shared USM
//   1111 xxxx xxxx xxxx   local memory
//   0000 xxxx xxxx xxxx   private memory
//   0zzz xxxx xxxx xxxx   chained
//
enum class HeapType { DeviceUSM, HostUSM, SharedUSM, Local };

inline const char *ToString(HeapType Type) {
  switch (Type) {
  case HeapType::DeviceUSM:
    return "Device USM";
  case HeapType::HostUSM:
    return "Host USM";
  case HeapType::SharedUSM:
    return "Shared USM";
  case HeapType::Local:
    return "Local Memory";
  default:
    return "Unknown Heap Type";
  }
}

class Origin {
public:
  //   static bool isValidId(uint32_t id) { return id != 0 && id !=
  //   (uint32_t)-1; }

  uint32_t raw_id() const { return raw_id_; }

  bool isHeapOrigin() const {
    return isDeviceUSMOrigin() || isHostUSMOrigin() || isSharedUSMOrigin() ||
           isLocalOrigin();
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

    assert(false && "Unknown heap type");
    return HeapType::DeviceUSM; // Default fallback, should never reach here
  }

  uint32_t getHeapId() const {
    switch (getHeapType()) {
    case HeapType::DeviceUSM:
      return getDeviceUSMId();
    case HeapType::HostUSM:
      return getHostUSMId();
    case HeapType::SharedUSM:
      return getSharedUSMId();
    case HeapType::Local:
      return getLocalId();
    default:
      assert(false && "Unknown heap type");
      return 0;
    }
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
    // 1111 xxxx xxxx xxxx
    return raw_id_ >> kLocalShift == kLocalBits;
  }

  bool isPrivateOrigin() const {
    // 0000 xxxx xxxx xxxx
    return (raw_id_ >> kDepthShift) == (1 << kDepthBits);
  }

  bool isChainedOrigin() const {
    // 0zzz xxxx xxxx xxxx, zzz != 000
    return (raw_id_ >> kDepthShift) > (1 << kDepthBits);
  }

  uint32_t getDeviceUSMId() const {
    assert(isDeviceUSMOrigin());
    return raw_id_ & kDeviceUSMIdMask;
  }

  uint32_t getHostUSMId() const {
    assert(isHostUSMOrigin());
    return raw_id_ & kHostUSMIdMask;
  }

  uint32_t getSharedUSMId() const {
    assert(isSharedUSMOrigin());
    return raw_id_ & kSharedUSMIdMask;
  }

  uint32_t getLocalId() const {
    assert(isLocalOrigin());
    return raw_id_ & kLocalIdMask;
  }

  uint32_t getPrivateId() const {
    assert(isPrivateOrigin());
    return raw_id_ & kChainedIdMask;
  }

  uint32_t getChainedId() const {
    assert(isChainedOrigin());
    return raw_id_ & kChainedIdMask;
  }

  // Returns the next origin in the chain and the current stack trace.
  // Origin getNextChainedOrigin(StackTrace *stack) const {
  //   assert(isChainedOrigin());
  //   uint32_t prev_id;
  //   uint32_t StackId = ChainedOriginDepotGet(getChainedId(), &prev_id);
  //   if (stack)
  //     *stack = StackDepotGet(StackId);
  //   return Origin(prev_id);
  // }

  // StackTrace getStackTraceForDeviceUSM() const {
  //   return StackDepotGet(getDeviceUSMId());
  // }

  // StackTrace getStackTraceForHostUSM() const {
  //   return StackDepotGet(getHostUSMId());
  // }

  // StackTrace getStackTraceForSharedUSM() const {
  //   return StackDepotGet(getSharedUSMId());
  // }

  // StackTrace getStackTraceForLocal() const {
  //   return StackDepotGet(getLocalId());
  // }

  // static Origin CreateStackOrigin(uint32_t id) {
  //   assert((id & kStackIdMask) == id);
  //   return Origin((1 << kHeapShift) | id);
  // }

  StackTrace getHeapStackTrace() const {
    assert(isHeapOrigin());
    uint32_t StackId = getHeapId();
    return StackDepotGet(StackId);
  }

  static Origin CreateHeapOrigin(StackTrace &Stack, HeapType Type) {
    uint32_t StackId = StackDepotPut(Stack);
    assert(StackId);
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
    default:
      assert(false && "Unknown heap type");
      return Origin(0); // Should never reach here
    }
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

  static const int kLocalBits = 15;
  static const int kLocalShift = 32 - 4;

  static const int kDepthBits = 3;
  static const int kDepthShift = 32 - kDepthBits - 1;

  static const uint32_t kDeviceUSMIdMask = ((uint32_t)-1) >>
                                           (32 - kDeviceUSMShift);
  static const uint32_t kHostUSMIdMask = ((uint32_t)-1) >> (32 - kHostUSMShift);
  static const uint32_t kSharedUSMIdMask = ((uint32_t)-1) >>
                                           (32 - kSharedUSMShift);
  static const uint32_t kLocalIdMask = ((uint32_t)-1) >> (32 - kLocalShift);
  static const uint32_t kChainedIdMask = ((uint32_t)-1) >> (32 - kDepthShift);
  static const uint32_t kStackIdMask = ((uint32_t)-1) >> (32 - kDepthShift);

  uint32_t raw_id_;

  explicit Origin(uint32_t raw_id) : raw_id_(raw_id) {}

  int depth() const {
    assert(isChainedOrigin());
    return (raw_id_ >> kDepthShift) & ((1 << kDepthBits) - 1);
  }

public:
  static const int kMaxDepth = (1 << kDepthBits) - 1;
};

} // namespace msan
} // namespace ur_sanitizer_layer
