// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: MIT
#ifndef UR_LEAK_CHECK_H
#define UR_LEAK_CHECK_H 1

#include "backtrace.hpp"
#include "ur_validation_layer.hpp"

#include <mutex>
#include <unordered_map>
#include <utility>

#define MAX_BACKTRACE_FRAMES 64

namespace validation_layer {

struct RefCountContext {
  private:
    struct RefRuntimeInfo {
        int64_t refCount;
        std::vector<BacktraceLine> backtrace;
    };

    enum RefCountUpdateType {
        REFCOUNT_CREATE,
        REFCOUNT_INCREASE,
        REFCOUNT_DECREASE,
    };

    std::mutex mutex;
    std::unordered_map<void *, struct RefRuntimeInfo> counts;

    void updateRefCount(void *ptr, enum RefCountUpdateType type) {
        std::unique_lock<std::mutex> ulock(mutex);

        auto it = counts.find(ptr);

        switch (type) {
        case REFCOUNT_CREATE:
            if (it == counts.end()) {
                counts[ptr] = {1, getCurrentBacktrace()};
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
                counts[ptr].refCount++;
            }
            break;
        case REFCOUNT_DECREASE:
            if (it == counts.end()) {
                counts[ptr] = {-1, getCurrentBacktrace()};
            } else {
                counts[ptr].refCount--;
            }

            if (counts[ptr].refCount < 0) {
                context.logger.error(
                    "Attempting to release nonexistent handle {}", ptr);
            }
            break;
        }

        context.logger.debug("Reference count for handle {} changed to {}", ptr,
                             counts[ptr].refCount);

        if (counts[ptr].refCount == 0) {
            counts.erase(ptr);
        }
    }

  public:
    void createRefCount(void *ptr) { updateRefCount(ptr, REFCOUNT_CREATE); }

    void incrementRefCount(void *ptr) {
        updateRefCount(ptr, REFCOUNT_INCREASE);
    }

    void decrementRefCount(void *ptr) {
        updateRefCount(ptr, REFCOUNT_DECREASE);
    }

    void clear() { counts.clear(); }

    void logInvalidReferences() {
        for (auto &[ptr, refRuntimeInfo] : counts) {
            context.logger.error("Retained {} reference(s) to handle {}",
                                 refRuntimeInfo.refCount, ptr);
            context.logger.error("Handle {} was recorded for first time here:",
                                 ptr);
            for (size_t i = 0; i < refRuntimeInfo.backtrace.size(); i++) {
                context.logger.error(refRuntimeInfo.backtrace[i].c_str());
            }
        }
    }

} refCountContext;

} // namespace validation_layer

#endif /* UR_LEAK_CHECK_H */
