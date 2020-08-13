//==-- circular_buffer_extended.hpp - Circular buffer with host accessor ---==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#pragma once

#include <detail/circular_buffer.hpp>
#include <detail/scheduler/commands.hpp>

#include <cstddef>
#include <deque>
#include <unordered_map>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

class CircularBufferExtended {
public:
  using GenericCommandsT = CircularBuffer<Command *>;
  using HostAccessorCommandsT = std::unordered_map<std::deque<Command *>>;
  using IfGenericIsFullF = std::function<void(Command *, GenericCommandsT &)>;

  template <bool IsConst>
  struct IteratorT;

  using value_type = Command *;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;

  using iterator = typename IteratorT<false>;
  using const_iterator = typename IteratorT<true>;

  explicit CircularBufferExtended(std::size_t Capacity, IfGenericIsFullF &&F);

  iterator begin() {
    return iterator{*this, MGenericCommands.begin()};
  }

  iterator end() {
    return iterator{*this, MHostAccessorCommands.end()};
  }

  const_iterator cbegin() const {
    return const_iteratorr{*this, MGenericCommands.begin()};
  }

  const_iterator cend() const {
    return const_iterator{*this, MHostAccessorCommands.end()};
  }

  void push_back(value_type Cmd);

  // erases range [first, last)
  void erase(iterator &first, iterator &last);

  // return number of removed elements
  size_t remove(value_type Cmd);

private:
  friend struct IteratorT<true>;
  friend struct IteratorT<false>;

  GenericCommandsT MGenericCommands;
  HostAccessorCommandsT MHostAccessorCommands;
  IfGenericIsFullF MIfGenericIsFull;

  void addHostAccessorCommand(EmptyCommand *Cmd);
  void addGenericCommand(value_type Cmd);


  template <bool IsConst, typename T>
  struct ConstRef;

  template<typename T> struct ConstRef<true, T> {
    using type = const T &;
  };

  template<typename T> struct ConstRef<false, T> {
    using type = T &;
  };

  template <bool IsConst>
  struct HostAccessorIt {
    using HostT = typename ConstRef<IsConst, HostAccessorCommandsT>::type;
    using MapItT = HostAccessorCommandsT::iterator;
    using ValueItT = HostAccessorCommandsT::mapped_type::iterator;

    HostT MHost;
    MapItT MMapIt;
    ValueItT MValueIt;

    HostAccessorIt(HostT Host, MapItT MapIt) : MHost{Host}, MMapIt{MapIt} {
      if (MMapIt != MHost.end())
        MValueIt = MMapIt->second.begin();
    }

    HostAccessorIt(const HostAccessorIt<IsConst> &Other)
      : MHost{Other.MHost}, MMapIt{Other.MMapIt}, MValueIt{Other.MValueIt} {}

    HostAccessorIt(HostAccessorIt<IsConst> &&Other)
      : MHost{Other.MHost}, MMapIt{Other.MMapIt}, MValueIt{Other.MValueIt} {}

    value_type operator*() {
      return *MValueIt;
    }

    value_type operator*() const {
      return *MValueIt;
    }

    // pre-increment
    HostAccessorIt<IsConst> &operator++() {
      increment();
      return *this;
    }

    // post-increment
    HostAccessorIt<IsConst> operator++(int) {
      HostAccessorIt<IsConst> Result = *this;
      increment();

      return Result;
    }

  private:
    // return true if in the very end or the increment happened
    inline bool incrementMapItIfNeeded() {
      if (MValueIt == MMapIt->second.end()) {
        ++MMapIt;

        if (MMapIt != MHost.end())
          MValueIt = MMapIt->second.begin();

        return true;
      }

      return false;
    }

    void increment() {
      if (MMapIt == MHost.end())
        return;

      if (incrementMapItIfNeeded())
        return;

      ++MValueIt;

      incrementMapItIfNeeded();
    }
  };

public:
  // iterate over generic commands in the first place and over host accessors
  // later on
  template <bool IsConst>
  struct IteratorT {
    using HostT = typename ConstRef<IsConst, CircularBufferExtended>::type;
    using GCItT = GenericCommandsT::iterator;
    using HACIt = HostAccessorIt;

    HostT MHost;
    GCItT MGCIt;
    HACItT MHACIt;

    bool MGenericIsActive;

    IteratorT(HostT &Host, GCItT &GCIt)
        : MHost(Host), MGCIt{std::move(GCIt)}, MGenericIsActive{true} {}

    IteratorT(HostT &Host, HACItT &HACIt)
        : MHost(Host), MHACIt{std::move(HACIt)}, MGenericIsActive{false} {}

    IteratorT(const IteratorT<IsConst> &Other)
        : MHost(Other.MHost), MGCIt(Other.MGCIt), MHACIt(Other.MHACIt),
          MGenericIsActive(Other.MGenericIsActive) {}

    IteratorT(IteratorT<IsConst> &&Other)
        : MHost(std::move(Other.MHost)), MGCIt(std::move(Other.MGCIt)),
          MHACIt(std::move(Other.MHACIt)),
          MGenericIsActive(Other.MGenericIsActive) {}

    bool operator==(const IteratorT &Rhs) const {
      return &MHost == &Rhs.MHost && MGenericIsActive == Rhs.MGenericIsActive &&
          ((MGenericIsActive && MGCIt == Rhs.MGCIt) ||
           (!MGenericIsActive && MHACIt == Rhs.MHACIt));
    }

    // pre-increment
    IteratorT<IsConst> &operator++() {
      increment();
      return *this;
    }

    // post-increment
    IteratorT<IsConst> operator++(int) {
      IteratorT<IsConst> Other(*this);
      increment();

      return Other;
    }

    value_type operator*() const {
      if (MGenericIsActive && MGCIt != MHost.MGenericCommands.end())
        return *MGCIt;

      if (!MGenericIsActive && MHACIt != MHost.MHostAccessorCommands.end())
        return *MHACIt;

      assert(false);

      return nullptr;
    }

  private:
    void increment() {
      if (MGenericIsActive) {
        if (MGCIt == MHost.MGenericCommands.end()) {
          MGenericIsActive = false;
          MHACIt = MHost.MHostAccessorCommands.begin();
          return;
        }

        ++MGCIt;
        return;
      }

      assert(MGCIt == MHost.MGenericCommands.end());

      if (MHACIt == MHost.MHostAccessorCommands.end())
        return;

      ++MHACIt;
    }
  };
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
