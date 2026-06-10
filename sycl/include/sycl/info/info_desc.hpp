//==------- info_desc.hpp - SYCL information descriptors -------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

// Umbrella that pulls the per-trait-class headers plus the ext-trait chain.
// Existing consumers of <sycl/info/info_desc.hpp> see the same set of trait
// declarations as before; per-class consumers should prefer the matching
// per-class header to avoid pulling unrelated trait families.

#include <sycl/info/context.hpp>
#include <sycl/info/device.hpp>
#include <sycl/info/event.hpp>
#include <sycl/info/kernel.hpp>
#include <sycl/info/platform.hpp>
#include <sycl/info/queue.hpp>

// Ext trait headers participate in is_*_info_desc SFINAE through the umbrella.
#include <sycl/ext/codeplay/experimental/max_registers_query.hpp>
#include <sycl/ext/intel/info/device.hpp>
#include <sycl/ext/intel/info/kernel.hpp>
#include <sycl/ext/oneapi/experimental/bindless_image_info.hpp>
#include <sycl/ext/oneapi/experimental/composite_device.hpp>
#include <sycl/ext/oneapi/experimental/device_architecture.hpp>
#include <sycl/ext/oneapi/experimental/forward_progress.hpp>
#include <sycl/ext/oneapi/experimental/kernel_queue_info.hpp>
#include <sycl/ext/oneapi/experimental/max_work_groups.hpp>
#include <sycl/ext/oneapi/info/device.hpp>
#include <sycl/ext/oneapi/matrix/query-types.hpp>
