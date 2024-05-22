// RUN: %clang_cc1 -internal-isystem %S/Inputs -fsycl-is-host -triple x86_64-unknown-linux-gnu -disable-llvm-passes -emit-llvm -o - %s | FileCheck %s

#include "sycl.hpp"

// Tests that [[intel::numbanks()]], [[intel::fpga_register]], [[intel::private_copies()]], [[intel::doublepump]], [[intel::singlepump]], [[intel::merge()]], [[intel::fpga_memory()]], [[intel::bank_bits()]], [[intel::force_pow2_depth()]], [[intel::max_replicates()]], [[intel::bankwidth()]], [[intel::simple_dual_port]] attributes are ignored on host code.

using namespace sycl::ext::oneapi;
using namespace sycl;

struct bar {
  [[intel::numbanks(2)]] device_global<int> nonconst_glob;
  [[intel::numbanks(4)]] const device_global<int> const_glob;
  [[intel::numbanks(8)]] unsigned int numbanks[64];

  [[intel::max_replicates(2)]] device_global<int> nonconst_glob1;
  [[intel::max_replicates(4)]] const device_global<int> const_glob1;
  [[intel::max_replicates(8)]] unsigned int max_rep[64];

  [[intel::force_pow2_depth(0)]] device_global<int> nonconst_glob2;
  [[intel::force_pow2_depth(0)]] const device_global<int> const_glob2;
  [[intel::force_pow2_depth(1)]] unsigned int force_dep[64];

  [[intel::bankwidth(2)]] device_global<int> nonconst_glob3;
  [[intel::bankwidth(4)]] const device_global<int> const_glob3;
  [[intel::bankwidth(16)]] unsigned int bankw[64];

  [[intel::simple_dual_port]] device_global<int> nonconst_glob4;
  [[intel::simple_dual_port]] const device_global<int> const_glob4;
  [[intel::simple_dual_port]] unsigned int simple[64];

  [[intel::fpga_memory]] device_global<int> nonconst_glob5;
  [[intel::fpga_memory("MLAB")]] const device_global<int> const_glob5;
  [[intel::fpga_memory("BLOCK_RAM")]] unsigned int mem_block_ram[32];

  [[intel::bank_bits(3, 4)]] device_global<int> nonconst_glob6;
  [[intel::bank_bits(4, 5)]] const device_global<int> const_glob6;
  [[intel::bank_bits(3, 4)]] unsigned int mem_block_bits[32];

  [[intel::fpga_register]] device_global<int> nonconst_glob7;
  [[intel::fpga_register]] const device_global<int> const_glob7;
  [[intel::fpga_register]] unsigned int reg;

  [[intel::singlepump]] device_global<int> nonconst_glob8;
  [[intel::singlepump]] const device_global<int> const_glob8;
  [[intel::singlepump]] unsigned int spump;

  [[intel::doublepump]] device_global<int> nonconst_glob9;
  [[intel::doublepump]] const device_global<int> const_glob9;
  [[intel::doublepump]] unsigned int dpump;

  [[intel::merge("mrg6", "depth")]] device_global<int> nonconst_glob10;
  [[intel::merge("mrg6", "depth")]] const device_global<int> const_glob10;
  [[intel::merge("mrg6", "width")]] unsigned int mergewidth;

  [[intel::private_copies(32)]] device_global<int> nonconst_glob11;
  [[intel::private_copies(8)]] const device_global<int> const_glob11;
  [[intel::private_copies(8)]] unsigned int pc;
};

[[intel::numbanks(4)]] device_global<int> nonconst_ignore;

[[intel::max_replicates(8)]] device_global<int> nonconst_ignore1;

[[intel::force_pow2_depth(0)]] device_global<int> nonconst_ignore2;

[[intel::bankwidth(2)]] device_global<int> nonconst_ignore3;

[[intel::simple_dual_port]] device_global<int> nonconst_ignore4;

[[intel::fpga_memory("MLAB")]] device_global<int> nonconst_ignore5;

[[intel::bank_bits(6, 7)]] device_global<int> nonconst_ignore6;

[[intel::fpga_register]] device_global<int> nonconst_ignore7;

[[intel::doublepump]] device_global<int> nonconst_ignore8;

[[intel::singlepump]] device_global<int> nonconst_ignore9;

[[intel::merge("mrg1", "depth")]] device_global<int> nonconst_ignore10;

[[intel::private_copies(16)]] device_global<int> nonconst_ignore11;

// CHECK-NOT: !private_copies
// CHECK-NOT: !singlepump
// CHECK-NOT: !doublepump
// CHECK-NOT: !force_pow2_depth
// CHECK-NOT: !max_replicates
// CHECK-NOT: !numbanks
// CHECK-NOT: !bank_bits
// CHECK-NOT: !bankwidth
// CHECK-NOT: !simple_dual_port
// CHECK-NOT: !merge
// CHECK-NOT: !fpga_memory
// CHECK-NOT: !fpga_register
