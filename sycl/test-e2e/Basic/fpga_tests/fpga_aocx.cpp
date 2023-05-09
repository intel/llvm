//==---- fpga_aocx.cpp - AOT compilation for fpga emulator dev with aocx ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: opencl-aot, accelerator

/// E2E test for AOCX creation/use/run for FPGA
// Produce an archive with device (AOCX) image. To avoid appending objects to
// leftover archives, remove one if exists.

// RUN: rm %t_image.a || true
// RUN: %clangxx -fsycl -fintelfpga -fsycl-link=image %S/Inputs/fpga_device.cpp -o %t_image.a
// Produce a host object
// RUN: %clangxx -fsycl -fintelfpga %S/Inputs/fpga_host.cpp -c -o %t.o

// AOCX with source
// RUN: %clangxx -fsycl -fintelfpga %S/Inputs/fpga_host.cpp %t_image.a -o %t_aocx_src.out
// AOCX with object
// RUN: %clangxx -fsycl -fintelfpga %t.o %t_image.a -o %t_aocx_obj.out
//
// RUN: %{run} %t_aocx_src.out
// RUN: %{run} %t_aocx_obj.out
