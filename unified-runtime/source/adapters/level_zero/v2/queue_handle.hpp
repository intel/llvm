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

#include "../common.hpp"
#include "queue_immediate_in_order.hpp"
#include "queue_immediate_out_of_order.hpp"
#include <ur_api.h>
#include <variant>

struct ur_queue_handle_t_ : ur::handle_base<ur::level_zero::ddi_getter> {
  using data_variant = std::variant<v2::ur_queue_immediate_in_order_t,
                                    v2::ur_queue_immediate_out_of_order_t>;
  data_variant queue_data;

  static constexpr uintptr_t queue_offset =
      sizeof(ur::handle_base<ur::level_zero::ddi_getter>);

  template <typename T, class... Args>
  ur_queue_handle_t_(std::in_place_type_t<T>, Args &&...args)
      : ur::handle_base<ur::level_zero::ddi_getter>(),
        queue_data(std::in_place_type<T>, std::forward<Args>(args)...) {
    assert(queue_offset ==
           (std::visit([](auto &q) { return reinterpret_cast<uintptr_t>(&q); },
                       queue_data) -
            reinterpret_cast<uintptr_t>(this)));
  }

  template <typename T, class... Args>
  static ur_queue_handle_t_ *create(Args &&...args) {
    return new ur_queue_handle_t_(std::in_place_type<T>,
                                  std::forward<Args>(args)...);
  }

  ur_queue_t_ &get() {
    return std::visit([&](auto &q) -> ur_queue_t_ & { return q; }, queue_data);
  }

  ur_result_t queueRetain() {
    return std::visit(
        [](auto &q) {
          q.RefCount.increment();
          return UR_RESULT_SUCCESS;
        },
        queue_data);
  }

  ur_result_t queueRelease() {
    return std::visit(
        [queueHandle = this](auto &q) {
          if (!q.RefCount.decrementAndTest())
            return UR_RESULT_SUCCESS;
          delete queueHandle;
          return UR_RESULT_SUCCESS;
        },
        queue_data);
  }
};
