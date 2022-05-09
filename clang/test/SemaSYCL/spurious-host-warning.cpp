// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify=host %s

// Test checks the attribute is silently ignored during host compilation
// where -fsycl-is-host is passed on cc1.

// expected-no-diagnostics

void foo()
{
  [[intel::doublepump]] unsigned int v_one[64]; // host-warning {{'doublepump' attribute ignored}}

  [[intel::fpga_memory]] unsigned int v_two[64]; // host-warning {{'fpga_memory' attribute ignored}}

  [[intel::fpga_register]] unsigned int v_three[64]; // host-warning {{'fpga_register' attribute ignored}}

  [[intel::singlepump]] unsigned int v_four[64]; // host-warning {{'singlepump' attribute ignored}}

  [[intel::bankwidth(4)]] unsigned int v_five[32]; // host-warning {{'bankwidth' attribute ignored}}

  [[intel::numbanks(8)]] unsigned int v_six[32]; // host-warning {{'numbanks' attribute ignored}}

  [[intel::private_copies(8)]] unsigned int v_seven[64]; // host-warning {{'private_copies' attribute ignored}}

  [[intel::merge("mrg1", "depth")]] unsigned int v_eight[64]; // host-warning {{'merge' attribute ignored}}

  [[intel::max_replicates(2)]] unsigned int v_nine[64]; // host-warning {{'max_replicates' attribute ignored}}

  [[intel::simple_dual_port]] unsigned int v_ten[64]; // host-warning {{'simple_dual_port' attribute ignored}}

  [[intel::bank_bits(2, 3, 4, 5)]] unsigned int v_eleven[64]; // host-warning {{'bank_bits' attribute ignored}}

  [[intel::use_stall_enable_clusters]] void func(); // host-warning {{'use_stall_enable_clusters' attribute ignored}}

  [[intel::max_global_work_dim(1)]] void func1(); // host-warning {{'max_global_work_dim' attribute ignored}}

  [[intel::scheduler_target_fmax_mhz(3)]] void func2(); // host-warning {{'scheduler_target_fmax_mhz' attribute ignored}}

  [[intel::kernel_args_restrict]] void func3(); // host-warning {{'kernel_args_restrict' attribute ignored}}

  [[intel::num_simd_work_items(12)]] void func4(); // host-warning {{'num_simd_work_items' attribute ignored}}

  [[intel::max_work_group_size(32, 32, 32)]] void func5(); // host-warning {{'max_work_group_size' attribute ignored}}

  [[intel::device_indirectly_callable]] void func6(); // host-warning {{'device_indirectly_callable' attribute ignored}}
}
