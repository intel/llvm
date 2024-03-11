// RUN: %{build} -o %t.out
// RUN: echo 1  | %{run} %t.out
// RUN: echo 10 | %{run} %t.out
// RUN: echo 20 | %{run} %t.out
// RUN: echo 30 | %{run} %t.out
// UNSUPPORTED: cuda || hip

// Simple test filling a private alloca and copying it back to an output
// accessor using a raw multi_ptr. This pointer checks struct allocation.

#include "Inputs/private_alloca_test.hpp"

constexpr sycl::specialization_id<std::size_t> size(10);

class value_and_sign {
public:
  value_and_sign() = default;
  constexpr explicit value_and_sign(int n)
      : value{static_cast<unsigned>(std::abs(n))}, no_less_than_zero(n >= 0) {}

  constexpr friend bool operator==(const value_and_sign &lhs, int rhs) {
    return lhs == value_and_sign(rhs);
  }

  constexpr friend bool operator==(int lhs, const value_and_sign &rhs) {
    return value_and_sign(lhs) == rhs;
  }

  constexpr friend bool operator==(const value_and_sign &lhs,
                                   const value_and_sign &rhs) {
    return lhs.no_less_than_zero == rhs.no_less_than_zero &&
           lhs.value == rhs.value;
  }

  constexpr value_and_sign &operator++() {
    inc();
    return *this;
  }

  constexpr value_and_sign operator++(int) {
    value_and_sign cpy = *this;
    inc();
    return cpy;
  }

  operator std::size_t() const {
    assert(no_less_than_zero && "Expecting value no less than zero");
    return value;
  }

private:
  constexpr void inc() { no_less_than_zero = ++value >= 0; }

  unsigned value;
  bool no_less_than_zero;
};

int main() { test<value_and_sign, size, sycl::access::decorated::no>(); }
