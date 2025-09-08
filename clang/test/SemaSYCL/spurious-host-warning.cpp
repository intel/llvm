// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify=device %s

// Test checks the attribute is silently ignored during host compilation
// where -fsycl-is-host is passed on cc1.

// expected-no-diagnostics

void foo()
{
  [[intel::doublepump]] unsigned int v_one[64]; // device-warning {{'intel::doublepump' attribute ignored}}

  [[intel::fpga_memory]] unsigned int v_two[64]; // device-warning {{'intel::fpga_memory' attribute ignored}}

  [[intel::fpga_register]] unsigned int v_three[64]; // device-warning {{'intel::fpga_register' attribute ignored}}

  [[intel::singlepump]] unsigned int v_four[64]; // device-warning {{'intel::singlepump' attribute ignored}}

  [[intel::bankwidth(4)]] unsigned int v_five[32]; // device-warning {{'intel::bankwidth' attribute ignored}}

  [[intel::numbanks(8)]] unsigned int v_six[32]; // device-warning {{'intel::numbanks' attribute ignored}}

  [[intel::private_copies(8)]] unsigned int v_seven[64]; // device-warning {{'intel::private_copies' attribute ignored}}

  [[intel::merge("mrg1", "depth")]] unsigned int v_eight[64]; // device-warning {{'intel::merge' attribute ignored}}

  [[intel::max_replicates(2)]] unsigned int v_nine[64]; // device-warning {{'intel::max_replicates' attribute ignored}}

  [[intel::simple_dual_port]] unsigned int v_ten[64]; // device-warning {{'intel::simple_dual_port' attribute ignored}}

  [[intel::bank_bits(2, 3, 4, 5)]] unsigned int v_eleven[64]; // device-warning {{'intel::bank_bits' attribute ignored}}

  [[intel::use_stall_enable_clusters]] void func(); // device-warning {{'intel::use_stall_enable_clusters' attribute ignored}}

  [[intel::max_global_work_dim(1)]] void func1(); // device-warning {{'intel::max_global_work_dim' attribute ignored}}

  [[intel::scheduler_target_fmax_mhz(3)]] void func2(); // device-warning {{'intel::scheduler_target_fmax_mhz' attribute ignored}}

  [[intel::kernel_args_restrict]] void func3(); // device-warning {{'intel::kernel_args_restrict' attribute ignored}}

  [[intel::num_simd_work_items(12)]] void func4(); // device-warning {{'intel::num_simd_work_items' attribute ignored}}

  [[intel::max_work_group_size(32, 32, 32)]] void func5(); // device-warning {{'intel::max_work_group_size' attribute ignored}}

  [[intel::device_indirectly_callable]] void func6(); // device-warning {{'intel::device_indirectly_callable' attribute ignored}}
}
