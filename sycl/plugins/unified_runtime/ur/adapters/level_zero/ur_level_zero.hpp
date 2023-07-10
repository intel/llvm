//===--------- ur_level_zero.hpp - Level Zero Adapter -----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===-----------------------------------------------------------------===//
#pragma once

#include <cassert>
#include <list>
#include <map>
#include <stdarg.h>
#include <string>
#include <unordered_map>
#include <vector>

#include <ur/ur.hpp>
#include <ur_api.h>
#include <ze_api.h>
#include <zes_api.h>

#include "common.hpp"
#include "context.hpp"
#include "device.hpp"
#include "event.hpp"
#include "image.hpp"
#include "kernel.hpp"
#include "memory.hpp"
#include "platform.hpp"
#include "program.hpp"
#include "queue.hpp"
#include "sampler.hpp"
#include "usm.hpp"
