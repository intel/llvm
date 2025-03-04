//===--------- memory.hpp - Level Zero Adapter ---------------------------===//
//
// Copyright (C) 2024 Intel Corporation
//
// Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
// Exceptions. See LICENSE.TXT
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <mutex>

template <typename T> struct locked {
public:
  locked(T *object, std::unique_lock<std::mutex> &&lock)
      : lock_(std::move(lock)) {
    object_ = object;
  }
  T *operator->() { return object_; }

private:
  std::unique_lock<std::mutex> lock_;
  T *object_;
};

template <typename T> struct lockable {
public:
  lockable(T &&object) : object_(std::move(object)) {}
  locked<T> lock() {
    std::unique_lock lock{mut_};
    return locked<T>(&object_, std::move(lock));
  }
  T &get_no_lock() { return object_; }

private:
  T object_;
  std::mutex mut_;
};
