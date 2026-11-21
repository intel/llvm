// REQUIRES: target-spir
// RUN: %{build} %{full_atomic64_flags} -o %t.out
// RUN: %if aspect-atomic64 %{ %{run} %t.out %} %else %{ echo "Skipped: requires atomic64" %}

#include "add.h"

int main() {
  queue q;
  add_test_orders_scopes<access::address_space::generic_space, char *,
                         ptrdiff_t>(q, 32);
}
