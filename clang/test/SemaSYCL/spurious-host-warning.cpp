// RUN: %clang_cc1 -fsycl-is-host -triple x86_64-pc-linux-gnu -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple x86_64-pc-linux-gnu -fsyntax-only -verify=device %s

// Test checks the attribute is silently ignored during host compilation
// where -fsycl-is-host is passed on cc1.

// expected-no-diagnostics

void foo()
{
  [[intel::kernel_args_restrict]] void func3(); // device-warning {{'intel::kernel_args_restrict' attribute ignored}}

  [[intel::device_indirectly_callable]] void func6(); // device-warning {{'intel::device_indirectly_callable' attribute ignored}}
}
