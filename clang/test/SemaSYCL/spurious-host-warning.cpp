// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify=device %s

// Test checks the attribute is silently ignored during host compilation
// where -fsycl-is-host is passed on cc1.

// expected-no-diagnostics

void foo()
{
  [[intel::max_global_work_dim(1)]] void func1(); // device-warning {{'intel::max_global_work_dim' attribute ignored}}

  [[intel::kernel_args_restrict]] void func3(); // device-warning {{'intel::kernel_args_restrict' attribute ignored}}

  [[intel::num_simd_work_items(12)]] void func4(); // device-warning {{'intel::num_simd_work_items' attribute ignored}}

  [[intel::max_work_group_size(32, 32, 32)]] void func5(); // device-warning {{'intel::max_work_group_size' attribute ignored}}

  [[intel::device_indirectly_callable]] void func6(); // device-warning {{'intel::device_indirectly_callable' attribute ignored}}
}
