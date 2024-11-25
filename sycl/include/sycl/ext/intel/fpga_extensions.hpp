//==----------- fpga_extensions.hpp --- SYCL FPGA Extensions ---------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once
#include <sycl/detail/defines_elementary.hpp>
#include <sycl/detail/stl_type_traits.hpp>
#include <sycl/ext/intel/experimental/fpga_annotated_properties.hpp>
#include <sycl/ext/intel/experimental/fpga_kernel_properties.hpp>
#include <sycl/ext/intel/experimental/fpga_lsu.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/fpga_datapath.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/fpga_mem.hpp>
#include <sycl/ext/intel/experimental/fpga_mem/properties.hpp>
#include <sycl/ext/intel/experimental/pipes.hpp>
#include <sycl/ext/intel/experimental/task_sequence.hpp>
#include <sycl/ext/intel/experimental/task_sequence_properties.hpp>
#include <sycl/ext/intel/fpga_device_selector.hpp>
#include <sycl/ext/intel/fpga_dsp_control.hpp>
#include <sycl/ext/intel/fpga_loop_fuse.hpp>
#include <sycl/ext/intel/fpga_lsu.hpp>
#include <sycl/ext/intel/fpga_reg.hpp>
#include <sycl/ext/intel/pipes.hpp>
