//==------------ sycl.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <sycl/__impl/ONEAPI/atomic.hpp>
#include <sycl/__impl/ONEAPI/experimental/builtins.hpp>
#include <sycl/__impl/ONEAPI/filter_selector.hpp>
#include <sycl/__impl/ONEAPI/function_pointer.hpp>
#include <sycl/__impl/ONEAPI/group_algorithm.hpp>
#include <sycl/__impl/ONEAPI/matrix/matrix.hpp>
#include <sycl/__impl/ONEAPI/reduction.hpp>
#include <sycl/__impl/ONEAPI/sub_group.hpp>
#include <sycl/__impl/accessor.hpp>
#include <sycl/__impl/aspects.hpp>
#include <sycl/__impl/atomic.hpp>
#include <sycl/__impl/backend.hpp>
#include <sycl/__impl/buffer.hpp>
#include <sycl/__impl/builtins.hpp>
#include <sycl/__impl/context.hpp>
#include <sycl/__impl/device.hpp>
#include <sycl/__impl/device_selector.hpp>
#include <sycl/__impl/event.hpp>
#include <sycl/__impl/exception.hpp>
#include <sycl/__impl/feature_test.hpp>
#include <sycl/__impl/functional.hpp>
#include <sycl/__impl/group.hpp>
#include <sycl/__impl/group_algorithm.hpp>
#include <sycl/__impl/group_local_memory.hpp>
#include <sycl/__impl/handler.hpp>
#include <sycl/__impl/id.hpp>
#include <sycl/__impl/image.hpp>
#include <sycl/__impl/item.hpp>
#include <sycl/__impl/kernel.hpp>
#include <sycl/__impl/kernel_bundle.hpp>
#include <sycl/__impl/kernel_handler.hpp>
#include <sycl/__impl/marray.hpp>
#include <sycl/__impl/multi_ptr.hpp>
#include <sycl/__impl/nd_item.hpp>
#include <sycl/__impl/nd_range.hpp>
#include <sycl/__impl/pipes.hpp>
#include <sycl/__impl/platform.hpp>
#include <sycl/__impl/pointers.hpp>
#include <sycl/__impl/program.hpp>
#include <sycl/__impl/properties/all_properties.hpp>
#include <sycl/__impl/queue.hpp>
#include <sycl/__impl/range.hpp>
#include <sycl/__impl/reduction.hpp>
#include <sycl/__impl/sampler.hpp>
#include <sycl/__impl/specialization_id.hpp>
#include <sycl/__impl/stream.hpp>
#include <sycl/__impl/sub_group.hpp>
#include <sycl/__impl/types.hpp>
#include <sycl/__impl/usm.hpp>
#include <sycl/__impl/version.hpp>

namespace sycl {
  using namespace __sycl_internal::__v1;
}
