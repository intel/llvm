/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM Exceptions.
 * See LICENSE.TXT
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_mock_helpers.hpp
 *
 */

#pragma once

#include "ur_api.h"
#include <atomic>
#include <cstddef>
#include <string>
#include <unordered_map>

namespace mock {

struct dummy_handle_t_ {
    dummy_handle_t_() {}
    std::atomic<size_t> refCounter = 1;
};

using dummy_handle_t = dummy_handle_t_ *;

// Allocates a dummy handle of type T with support for reference counting.
// The handle has to be deallocated using 'releaseDummyHandle'.
template <class T> inline T createDummyHandle() {
    dummy_handle_t DummyHandlePtr = new dummy_handle_t_();
    return reinterpret_cast<T>(DummyHandlePtr);
}

// Decrement reference counter for the handle and deallocates it if the
// reference counter becomes zero
template <class T> inline void releaseDummyHandle(T Handle) {
    auto DummyHandlePtr = reinterpret_cast<dummy_handle_t>(Handle);
    const size_t NewValue = --DummyHandlePtr->refCounter;
    if (NewValue == 0) {
        delete DummyHandlePtr;
    }
}

// Increment reference counter for the handle
template <class T> inline void retainDummyHandle(T Handle) {
    auto DummyHandlePtr = reinterpret_cast<dummy_handle_t>(Handle);
    ++DummyHandlePtr->refCounter;
}

struct callbacks_t {
    void set_before_callback(std::string name, ur_mock_callback_t callback) {
        beforeCallbacks[name] = callback;
    }

    ur_mock_callback_t get_before_callback(std::string name) {
        auto callback = beforeCallbacks.find(name);

        if (callback != beforeCallbacks.end()) {
            return callback->second;
        }
        return nullptr;
    }

    void set_replace_callback(std::string name, ur_mock_callback_t callback) {
        replaceCallbacks[name] = callback;
    }

    ur_mock_callback_t get_replace_callback(std::string name) {
        auto callback = replaceCallbacks.find(name);

        if (callback != replaceCallbacks.end()) {
            return callback->second;
        }
        return nullptr;
    }

    void set_after_callback(std::string name, ur_mock_callback_t callback) {
        afterCallbacks[name] = callback;
    }

    ur_mock_callback_t get_after_callback(std::string name) {
        auto callback = afterCallbacks.find(name);

        if (callback != afterCallbacks.end()) {
            return callback->second;
        }
        return nullptr;
    }

    void resetCallbacks() {
        beforeCallbacks.clear();
        replaceCallbacks.clear();
        afterCallbacks.clear();
    }

  private:
    std::unordered_map<std::string, ur_mock_callback_t> beforeCallbacks;
    std::unordered_map<std::string, ur_mock_callback_t> replaceCallbacks;
    std::unordered_map<std::string, ur_mock_callback_t> afterCallbacks;
};

extern callbacks_t callbacks;

} // namespace mock
