/*
 *
 * Copyright (C) 2022-2023 Intel Corporation
 *
 * Part of the Unified-Runtime Project, under the Apache License v2.0 with LLVM
 * Exceptions. See LICENSE.TXT
 *
 * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 * @file ur_object.hpp
 *
 */

#ifndef UR_OBJECT_H
#define UR_OBJECT_H 1

#include "ur_ddi.h"
#include "ur_util.hpp"

//////////////////////////////////////////////////////////////////////////
struct dditable_t {
  ur_dditable_t ur;
};

//////////////////////////////////////////////////////////////////////////
template <typename _handle_t> class object_t {
public:
  using handle_t = _handle_t;

  handle_t handle;
  dditable_t *dditable;

  object_t() = delete;

  object_t(handle_t _handle, dditable_t *_dditable)
      : handle(_handle), dditable(_dditable) {}

  ~object_t() = default;
};

#endif /* UR_OBJECT_H */
