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

/*
  lockable<T> wraps T class object in exclusive access lock, similar to one used
  in rust

  construction:
    lockable<X> l(arguments, to, construct, X);

  access without synchronization:
    X* obj_ptr = l.get_no_lock();
    obj_ptr->print_name();

  exclusive access to object kept in l:
    // as long as lock exists, thread has exclusive access to underlaying object
    locked<X> lock = l.lock();
    // that object is accessed through ->() operator on lock object
    lock->print_name();
*/

template <typename T> struct lockable {
public:
  template <typename... Args>
  lockable(Args &&...args) : object_(std::forward<Args>(args)...) {}
  locked<T> lock() {
    std::unique_lock lock{mut_};
    return locked<T>(&object_, std::move(lock));
  }
  template <typename Base> locked<Base> lock() {
    std::unique_lock lock{mut_};
    return locked<Base>(&object_, std::move(lock));
  }
  T *get_no_lock() { return &object_; }

private:
  T object_;
  std::mutex mut_;
};
