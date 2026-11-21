// REQUIRES: target-spir
// RUN: %{build} %{full_atomic64_flags} -o %t.out
// RUN: %if aspect-atomic64 %{ %{run} %t.out %} %else %{ echo "Skipped: requires atomic64" %}

#include "sub.h"

int main() {
  queue q;
  sub_test_orders_scopes<access::address_space::global_space, char *,
                         ptrdiff_t>(q, 32);
}
