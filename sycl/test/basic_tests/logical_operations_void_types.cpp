
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <cassert>
#include <sycl/functional.hpp>
#include <type_traits>


const auto logicalAndVoid = sycl::logical_and<void>();
const auto logicalOrVoid = sycl::logical_or<void>();

// expected-error@+1 {{}}
static_assert(std::is_same_v<decltype(logicalAndVoid(static_cast<void>(1), static_cast<void>(2))), bool> == true);
// expected-error@+1 {{}}
static_assert(std::is_same_v<decltype(logicalOrVoid(static_cast<void>(1), static_cast<void>(2))), bool> == true);
