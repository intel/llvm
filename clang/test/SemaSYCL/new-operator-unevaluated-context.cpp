// RUN: %clang_cc1 -fsycl-is-device -internal-isystem %S/Inputs -verify -fsyntax-only -std=c++20 -triple spir64 -Wno-unused-value %s

// Tests that the "SYCL kernel cannot allocate storage" diagnostic is not
// emitted in unevaluated contexts such as C++20 concepts, requires clauses,
// decltype, sizeof, etc.

#include "sycl.hpp"

namespace std {
class type_info;
typedef __typeof__(sizeof(int)) size_t;
} // namespace std

// Stub implementation for operator new
void *operator new(std::size_t) {
  return reinterpret_cast<void *>(1);
}


template <typename T>
concept Allocatable = requires {
  // No error expected - this is in an unevaluated context
  ::new T;
};

template <typename T>
concept AllocatableArray = requires {
  // No error expected - this is in an unevaluated context
  ::new T[10];
};

static_assert(Allocatable<int>);
static_assert(AllocatableArray<int>);


template <typename T>
  requires requires { ::new T; }
void test_requires_clause() {}

template <typename T>
  requires requires { ::new T[5]; }
void test_requires_clause_array() {}


void test_decltype() {
  // No error expected - inside decltype
  decltype(new int) ptr1;
  decltype(new int[10]) ptr2;
}


void test_sizeof() {
  // No error expected - inside sizeof
  constexpr std::size_t s1 = sizeof(new int);
  constexpr std::size_t s2 = sizeof(new int[10]);
}


void test_alignof() {
  // No error expected - inside alignof of the result type
  constexpr std::size_t a1 = alignof(int*);
  using PtrType = decltype(new int);
  constexpr std::size_t a2 = alignof(PtrType);
}


void test_noexcept() noexcept(noexcept(new int)) {}


// Usage in kernel should emit diagnostic
void actual_allocation() {
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  int *ptr = new int;
}

void actual_array_allocation() {
  // expected-error@+1 {{SYCL kernel cannot allocate storage}}
  int *ptr = new int[10];
}


int main() {
  sycl::queue q;

  q.submit([&](sycl::handler &h) {
    h.single_task<class TestConcepts>([]() {
      // These should all work without errors
      static_assert(Allocatable<int>);
      static_assert(AllocatableArray<double>);

      test_requires_clause<int>();
      test_requires_clause_array<float>();

      test_decltype();
      test_sizeof();
      test_alignof();
      test_noexcept();
    });
  });

  q.submit([&](sycl::handler &h) {
    h.single_task<class TestActualAlloc>([&]() {
      // expected-note@Inputs/sycl.hpp:322 {{called by 'kernel_single_task<TestActualAlloc}}
      actual_allocation(); // expected-note {{called by 'operator()'}}
    });
  });

  q.submit([&](sycl::handler &h) {
    h.single_task<class TestActualArrayAlloc>([&]() {
      // expected-note@Inputs/sycl.hpp:322 {{called by 'kernel_single_task<TestActualArrayAlloc}}
      actual_array_allocation(); // expected-note {{called by 'operator()'}}
    });
  });

  return 0;
}
