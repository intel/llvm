/*
 *
 * Copyright (C) 2024 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file queue_handle.hpp
 *
 */

#pragma once

#include "queue_immediate_in_order.hpp"
#include <ur_api.h>
#include <variant>

struct ur_queue_handle_t_ {
  using data_variant = std::variant<v2::ur_queue_immediate_in_order_t>;
  data_variant queue_data;

  template <typename T, class... Args>
  static ur_queue_handle_t_ *create(Args &&...args) {
    return new ur_queue_handle_t_{data_variant{std::in_place_type<T>, args...}};
  }

  ur_queue_t_ &get() {
    return std::visit([&](auto &q) -> ur_queue_t_ & { return q; }, queue_data);
  }
};
