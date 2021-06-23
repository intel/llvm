//==------------ sycl.hpp - SYCL standard header file ----------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <CL/sycl/ONEAPI/atomic.hpp>
#include <CL/sycl/ONEAPI/experimental/builtins.hpp>
#include <CL/sycl/ONEAPI/filter_selector.hpp>
#include <CL/sycl/ONEAPI/function_pointer.hpp>
#include <CL/sycl/ONEAPI/group_algorithm.hpp>
#include <CL/sycl/ONEAPI/matrix/matrix.hpp>
#include <CL/sycl/ONEAPI/reduction.hpp>
#include <CL/sycl/ONEAPI/sub_group.hpp>
#include <CL/sycl/accessor.hpp>
#include <CL/sycl/aspects.hpp>
#include <CL/sycl/atomic.hpp>
#include <CL/sycl/backend.hpp>
#include <CL/sycl/buffer.hpp>
#include <CL/sycl/builtins.hpp>
#include <CL/sycl/context.hpp>
#include <CL/sycl/device.hpp>
#include <CL/sycl/device_selector.hpp>
#include <CL/sycl/event.hpp>
#include <CL/sycl/exception.hpp>
#include <CL/sycl/feature_test.hpp>
#include <CL/sycl/functional.hpp>
#include <CL/sycl/group.hpp>
#include <CL/sycl/group_algorithm.hpp>
#include <CL/sycl/group_local_memory.hpp>
#include <CL/sycl/handler.hpp>
#include <CL/sycl/id.hpp>
#include <CL/sycl/image.hpp>
#include <CL/sycl/item.hpp>
#include <CL/sycl/kernel.hpp>
#include <CL/sycl/kernel_bundle.hpp>
#include <CL/sycl/kernel_handler.hpp>
#include <CL/sycl/marray.hpp>
#include <CL/sycl/multi_ptr.hpp>
#include <CL/sycl/nd_item.hpp>
#include <CL/sycl/nd_range.hpp>
#include <CL/sycl/pipes.hpp>
#include <CL/sycl/platform.hpp>
#include <CL/sycl/pointers.hpp>
#include <CL/sycl/program.hpp>
#include <CL/sycl/properties/all_properties.hpp>
#include <CL/sycl/queue.hpp>
#include <CL/sycl/range.hpp>
#include <CL/sycl/reduction.hpp>
#include <CL/sycl/sampler.hpp>
#include <CL/sycl/specialization_id.hpp>
#include <CL/sycl/stream.hpp>
#include <CL/sycl/sub_group.hpp>
#include <CL/sycl/types.hpp>
#include <CL/sycl/usm.hpp>
#include <CL/sycl/version.hpp>
