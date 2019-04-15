//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: libcpp-no-exceptions, libcpp-no-if-constexpr
// MODULES_DEFINES: _LIBCPP_DEBUG=1
// MODULES_DEFINES: _LIBCPP_DEBUG_USE_EXCEPTIONS

// Can't test the system lib because this test enables debug mode
// UNSUPPORTED: with_system_cxx_lib

// test container debugging

#define _LIBCPP_DEBUG 1
#define _LIBCPP_DEBUG_USE_EXCEPTIONS
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <cassert>
#include "debug_mode_helper.h"

using namespace IteratorDebugChecks;

template <class Container, ContainerType CT>
struct UnorderedContainerChecks : BasicContainerChecks<Container, CT> {
  using Base = BasicContainerChecks<Container, CT>;
  using value_type = typename Container::value_type;
  using iterator = typename Container::iterator;
  using const_iterator = typename Container::const_iterator;
  using traits = std::iterator_traits<iterator>;
  using category = typename traits::iterator_category;

  using Base::makeContainer;
public:
  static void run() {
    Base::run();
    try {
     // FIXME
    } catch (...) {
      assert(false && "uncaught debug exception");
    }
  }
private:

};

int main(int, char**)
{
  using SetAlloc = test_allocator<int>;
  using MapAlloc = test_allocator<std::pair<const int, int>>;
  {
    UnorderedContainerChecks<
        std::unordered_map<int, int, std::hash<int>, std::equal_to<int>, MapAlloc>,
        CT_UnorderedMap>::run();
    UnorderedContainerChecks<
        std::unordered_set<int, std::hash<int>, std::equal_to<int>, SetAlloc>,
        CT_UnorderedSet>::run();
    UnorderedContainerChecks<
        std::unordered_multimap<int, int, std::hash<int>, std::equal_to<int>, MapAlloc>,
        CT_UnorderedMultiMap>::run();
    UnorderedContainerChecks<
        std::unordered_multiset<int, std::hash<int>, std::equal_to<int>, SetAlloc>,
        CT_UnorderedMultiSet>::run();
  }

  return 0;
}
