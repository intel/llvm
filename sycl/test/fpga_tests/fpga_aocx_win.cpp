//==--- fpga_aocx_win.cpp - AOT compilation for fpga using aoc with aocx ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: aoc, accelerator
// REQUIRES: system-windows

/// E2E test for AOCX creation/use/run for FPGA
// Produce an archive with device (AOCX) image
// RUN: %clang_cl -fsycl -fintelfpga -fsycl-link=image %S/Inputs/fpga_device.cpp -o %t_image.lib
// Produce a host object
// RUN: %clang_cl -fsycl -fintelfpga -DHOST_PART %S/Inputs/fpga_host.cpp -c -o %t.obj

// AOCX with source
// RUN: %clang_cl -fsycl -fintelfpga -DHOST_PART %S/Inputs/fpga_host.cpp %t_image.lib -o %t_aocx_src.out
// AOCX with object
// RUN: %clang_cl -fsycl -fintelfpga %t.obj %t_image.lib -o %t_aocx_obj.out
//
// RUN: env SYCL_DEVICE_TYPE=ACC %t_aocx_src.out
// RUN: env SYCL_DEVICE_TYPE=ACC %t_aocx_obj.out
