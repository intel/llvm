//===- VariadicIterator.cpp
//------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <detail/helpers.hpp>

#include <gtest/gtest.h>

#include <memory>
#include <stdexcept>
#include <utility>
#include <variant>

namespace {

struct TestSyclObject {
  std::shared_ptr<int> impl;
};

struct ThrowingIterator {
  static bool ThrowOnMove;

  int *Ptr;
  bool ThrowOnUse = false;

  explicit ThrowingIterator(int *Ptr, bool ThrowOnUse = false)
      : Ptr(Ptr), ThrowOnUse(ThrowOnUse) {}
  ThrowingIterator(const ThrowingIterator &) = default;
  ThrowingIterator(ThrowingIterator &&Other)
      : Ptr(Other.Ptr), ThrowOnUse(Other.ThrowOnUse) {
    if (ThrowOnMove)
      throw std::runtime_error("iterator move");
  }
  ThrowingIterator &operator=(const ThrowingIterator &) = default;
  ThrowingIterator &operator=(ThrowingIterator &&) = default;

  int &operator*() const {
    if (ThrowOnUse)
      throw std::runtime_error("iterator dereference");
    return *Ptr;
  }
  ThrowingIterator &operator++() {
    if (ThrowOnUse)
      throw std::runtime_error("iterator increment");
    ++Ptr;
    return *this;
  }
  bool operator==(const ThrowingIterator &Other) const {
    if (ThrowOnUse)
      throw std::runtime_error("iterator comparison");
    return Ptr == Other.Ptr;
  }
  bool operator!=(const ThrowingIterator &Other) const {
    return !(*this == Other);
  }
};

bool ThrowingIterator::ThrowOnMove = false;

using Iterator =
    sycl::detail::variadic_iterator<TestSyclObject, int *, ThrowingIterator>;

TEST(VariadicIteratorTest, PropagatesUnderlyingIteratorExceptions) {
  int Value = 42;
  Iterator It(ThrowingIterator(&Value, true));

  EXPECT_THROW(*It, std::runtime_error);
  EXPECT_THROW(++It, std::runtime_error);
  EXPECT_THROW((void)(It == It), std::runtime_error);
  EXPECT_THROW((void)(It != It), std::runtime_error);
}

TEST(VariadicIteratorTest, PropagatesBadVariantAccess) {
  int Value = 42;
  Iterator Target(&Value);
  Iterator Source{ThrowingIterator(&Value)};

  // Changing alternatives can leave a variant valueless if construction fails.
  ThrowingIterator::ThrowOnMove = true;
  EXPECT_THROW(Target = std::move(Source), std::runtime_error);
  ThrowingIterator::ThrowOnMove = false;

  EXPECT_THROW(*Target, std::bad_variant_access);
  EXPECT_THROW(++Target, std::bad_variant_access);
}

} // namespace
