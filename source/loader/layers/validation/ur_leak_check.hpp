// Copyright (C) 2023-2024 Intel Corporation
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
// See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef UR_LEAK_CHECK_H
#define UR_LEAK_CHECK_H 1

#include "backtrace.hpp"
#include "ur_validation_layer.hpp"

#include <mutex>
#include <typeindex>
#include <unordered_map>
#include <utility>

#define MAX_BACKTRACE_FRAMES 64

namespace ur_validation_layer {

struct RefCountContext {
  private:
    struct RefRuntimeInfo {
        int64_t refCount;
        std::type_index type;
        std::vector<BacktraceLine> backtrace;

        RefRuntimeInfo(int64_t refCount, std::type_index type,
                       std::vector<BacktraceLine> backtrace)
            : refCount(refCount), type(type), backtrace(backtrace) {}
    };

    enum RefCountUpdateType {
        REFCOUNT_CREATE_OR_INCREASE,
        REFCOUNT_CREATE,
        REFCOUNT_INCREASE,
        REFCOUNT_DECREASE,
    };

    std::mutex mutex;
    std::unordered_map<void *, struct RefRuntimeInfo> counts;
    int64_t adapterCount = 0;

    template <typename T>
    void updateRefCount(T handle, enum RefCountUpdateType type,
                        bool isAdapterHandle = false) {
        std::unique_lock<std::mutex> ulock(mutex);

        void *ptr = static_cast<void *>(handle);
        auto it = counts.find(ptr);

        switch (type) {
        case REFCOUNT_CREATE_OR_INCREASE:
            if (it == counts.end()) {
                std::tie(it, std::ignore) = counts.emplace(
                    ptr, RefRuntimeInfo{1, std::type_index(typeid(handle)),
                                        getCurrentBacktrace()});
                if (isAdapterHandle) {
                    adapterCount++;
                }
            } else {
                it->second.refCount++;
            }
            break;
        case REFCOUNT_CREATE:
            if (it == counts.end()) {
                std::tie(it, std::ignore) = counts.emplace(
                    ptr, RefRuntimeInfo{1, std::type_index(typeid(handle)),
                                        getCurrentBacktrace()});
            } else {
                context.logger.error("Handle {} already exists", ptr);
                return;
            }
            break;
        case REFCOUNT_INCREASE:
            if (it == counts.end()) {
                context.logger.error(
                    "Attempting to retain nonexistent handle {}", ptr);
                return;
            } else {
                it->second.refCount++;
            }
            break;
        case REFCOUNT_DECREASE:
            if (it == counts.end()) {
                std::tie(it, std::ignore) = counts.emplace(
                    ptr, RefRuntimeInfo{-1, std::type_index(typeid(handle)),
                                        getCurrentBacktrace()});
            } else {
                it->second.refCount--;
            }

            if (it->second.refCount < 0) {
                context.logger.error(
                    "Attempting to release nonexistent handle {}", ptr);
            } else if (it->second.refCount == 0 && isAdapterHandle) {
                adapterCount--;
            }
            break;
        }

        context.logger.debug("Reference count for handle {} changed to {}", ptr,
                             it->second.refCount);

        if (it->second.refCount == 0) {
            counts.erase(ptr);
        }

        // No more active adapters, so any references still held are leaked
        if (adapterCount == 0) {
            logInvalidReferences();
            clear();
        }
    }

  public:
    template <typename T> void createRefCount(T handle) {
        updateRefCount<T>(handle, REFCOUNT_CREATE);
    }

    template <typename T>
    void incrementRefCount(T handle, bool isAdapterHandle = false) {
        updateRefCount(handle, REFCOUNT_INCREASE, isAdapterHandle);
    }

    template <typename T>
    void decrementRefCount(T handle, bool isAdapterHandle = false) {
        updateRefCount(handle, REFCOUNT_DECREASE, isAdapterHandle);
    }

    template <typename T>
    void createOrIncrementRefCount(T handle, bool isAdapterHandle = false) {
        updateRefCount(handle, REFCOUNT_CREATE_OR_INCREASE, isAdapterHandle);
    }

    void clear() { counts.clear(); }

    template <typename T> bool isReferenceValid(T handle) {
        auto it = counts.find(static_cast<void *>(handle));
        if (it == counts.end() || it->second.refCount < 1) {
            return false;
        }

        return (it->second.type == std::type_index(typeid(handle)));
    }

    void logInvalidReferences() {
        for (auto &[ptr, refRuntimeInfo] : counts) {
            context.logger.error("Retained {} reference(s) to handle {}",
                                 refRuntimeInfo.refCount, ptr);
            context.logger.error("Handle {} was recorded for first time here:",
                                 ptr);
            for (size_t i = 0; i < refRuntimeInfo.backtrace.size(); i++) {
                context.logger.error("#{} {}", i,
                                     refRuntimeInfo.backtrace[i].c_str());
            }
        }
    }

    void logInvalidReference(void *ptr) {
        context.logger.error("There are no valid references to handle {}", ptr);
    }

} refCountContext;

} // namespace ur_validation_layer

#endif /* UR_LEAK_CHECK_H */
