
// RUN: %clangxx -fsycl -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s
// RUN: %clangxx -fsycl -fpreview-breaking-changes -fsyntax-only -Xclang -verify -Xclang -verify-ignore-unexpected=note %s

#include <sycl/functional.hpp>
#include <type_traits>

const auto logicalAndVoid = sycl::logical_and<void>();
const auto logicalOrVoid = sycl::logical_or<void>();

// expected-error@+1 {{}}
logicalAndVoid(static_cast<void>(1), static_cast<void>(2));
// expected-error@+1 {{}}
logicalOrVoid(static_cast<void>(1), static_cast<void>(2));
