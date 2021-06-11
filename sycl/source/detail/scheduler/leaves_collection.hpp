//==---- leaves_collection.hpp - Container for leaves of execution graph ---==//
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
#include <list>
#include <unordered_map>
#include <utility>

__SYCL_INLINE_NAMESPACE(cl) {
namespace sycl {
namespace detail {

struct MemObjRecord;

/// A wrapper for CircularBuffer class along with collection for host accessor's
/// EmptyCommands. This class is introduced to overcome the problem with a lot
/// of host accessors for the same memory object. The problem arises even when
/// all the host accessors are read-only and their ranges intersect somehow.
///
/// Inside the class there is a classical CircularBuffer for generic commands.
/// Also, it contains proper data structures for storing host accessors'
/// EmptyCommands. Cross-referencing data structures are employed for
/// quick enough navigation amongst stored EmptyCommands.
/// IteratorT subclass allows for iterating and dereferencing. Though, it's not
/// guaranteed to work with std::remove as host accessors' commands are stored
/// in a map. Hence, the LeavesCollection class provides a viable solution
/// with its own remove method.
class LeavesCollection {
public:
  using GenericCommandsT = CircularBuffer<Command *>;
  using HostAccessorCommandsT = std::list<EmptyCommand *>;
  using EnqueueListT = std::vector<Command *>;

  // Make first command depend on the second
  using AllocateDependencyF =
      std::function<void(Command *, Command *, MemObjRecord *, EnqueueListT &)>;

  template <bool IsConst> class IteratorT;

  using value_type = Command *;
  using pointer = value_type *;
  using const_pointer = const value_type *;
  using reference = value_type &;
  using const_reference = const value_type &;

  using iterator = IteratorT<false>;
  using const_iterator = IteratorT<true>;

  LeavesCollection(MemObjRecord *Record, std::size_t GenericCommandsCapacity,
                   AllocateDependencyF AllocateDependency)
      : MRecord{Record}, MGenericCommands{GenericCommandsCapacity},
        MAllocateDependency{std::move(AllocateDependency)} {}

  iterator begin() {
    if (MGenericCommands.empty())
      return iterator{*this, beginHostAccessor()};

    return iterator{*this, MGenericCommands.begin()};
  }

  iterator end() { return iterator{*this, endHostAccessor()}; }

  const_iterator cbegin() const {
    if (MGenericCommands.empty())
      return const_iterator{*this, beginHostAccessor()};

    return const_iterator{*this, MGenericCommands.begin()};
  }

  const_iterator cend() const {
    return const_iterator{*this, endHostAccessor()};
  }

  /// Returns true if insertion took place. Returns false otherwise.
  bool push_back(value_type Cmd, EnqueueListT &ToEnqueue);

  /// Replacement for std::remove with subsequent call to erase(newEnd, end()).
  /// This function is introduced here due to complexity of iterator.
  /// \returns number of removed elements
  size_t remove(value_type Cmd);

  std::vector<value_type> toVector() const;

  size_t genericCommandsCapacity() const {
    return MGenericCommands.capacity();
  };

  const GenericCommandsT &getGenericCommands() const {
    return MGenericCommands;
  }

  const HostAccessorCommandsT getHostAccessorCommands() const {
    return MHostAccessorCommands;
  }

private:
  template <bool IsConst, typename T> struct Iterator;

  template <typename T> struct Iterator<true, T> {
    using type = typename T::const_iterator;
  };

  template <typename T> struct Iterator<false, T> {
    using type = typename T::iterator;
  };

  using HostAccessorCommandSingleXRefT =
      typename HostAccessorCommandsT::iterator;
  using HostAccessorCommandsXRefT =
      std::unordered_map<EmptyCommand *, HostAccessorCommandSingleXRefT>;

  MemObjRecord *MRecord;
  GenericCommandsT MGenericCommands;
  HostAccessorCommandsT MHostAccessorCommands;
  HostAccessorCommandsXRefT MHostAccessorCommandsXRef;

  AllocateDependencyF MAllocateDependency;

  bool addGenericCommand(value_type Cmd, EnqueueListT &ToEnqueue);
  bool addHostAccessorCommand(EmptyCommand *Cmd, EnqueueListT &ToEnqueue);

  // inserts a command to the end of list for its mem object
  void insertHostAccessorCommand(EmptyCommand *Cmd);
  // returns number of removed elements
  size_t eraseHostAccessorCommand(EmptyCommand *Cmd);

  typename Iterator<false, HostAccessorCommandsT>::type beginHostAccessor() {
    return MHostAccessorCommands.begin();
  }

  typename Iterator<true, HostAccessorCommandsT>::type
  beginHostAccessor() const {
    return MHostAccessorCommands.begin();
  }

  typename Iterator<false, HostAccessorCommandsT>::type endHostAccessor() {
    return MHostAccessorCommands.end();
  }

  typename Iterator<true, HostAccessorCommandsT>::type endHostAccessor() const {
    return MHostAccessorCommands.end();
  }

  // for access to struct Ref.
  friend class IteratorT<true>;
  friend class IteratorT<false>;

  template <bool IsConst, typename T> struct Ref;
  template <typename T> struct Ref<true, T> { using type = const T &; };
  template <typename T> struct Ref<false, T> { using type = T &; };

  template <bool IsConst, typename T> struct Ptr;
  template <typename T> struct Ptr<true, T> { using type = const T *; };
  template <typename T> struct Ptr<false, T> { using type = T *; };

public:
  // iterate over generic commands in the first place and over host accessors
  // later on
  template <bool IsConst> class IteratorT {
  public:
    using HostT = typename Ref<IsConst, LeavesCollection>::type;
    using GCItT = typename Iterator<IsConst, GenericCommandsT>::type;
    using HACItT = typename Iterator<IsConst, HostAccessorCommandsT>::type;

  private:
    HostT MHost;
    GCItT MGCIt;
    HACItT MHACIt;

    bool MGenericIsActive;

    IteratorT(HostT Host, GCItT GCIt, HACItT HACIt, bool GenericIsActive)
        : MHost(Host), MGCIt(GCIt), MHACIt(HACIt),
          MGenericIsActive(GenericIsActive) {}

  public:
    IteratorT(HostT Host, GCItT GCIt)
        : IteratorT(Host, std::move(GCIt), Host.beginHostAccessor(), true) {}

    IteratorT(HostT Host, HACItT HACIt)
        : IteratorT(Host, Host.MGenericCommands.end(), std::move(HACIt),
                    false) {}

    IteratorT(const IteratorT<IsConst> &Other)
        : MHost{Other.MHost}, MGCIt(Other.MGCIt), MHACIt(Other.MHACIt),
          MGenericIsActive(Other.MGenericIsActive) {}

    IteratorT(IteratorT<IsConst> &&Other)
        : MHost{Other.MHost}, MGCIt(std::move(Other.MGCIt)),
          MHACIt(std::move(Other.MHACIt)),
          MGenericIsActive(Other.MGenericIsActive) {}

  public:
    bool operator==(const IteratorT<IsConst> &Rhs) const {
      return &MHost == &Rhs.MHost && MGenericIsActive == Rhs.MGenericIsActive &&
             ((MGenericIsActive && MGCIt == Rhs.MGCIt) ||
              (!MGenericIsActive && MHACIt == Rhs.MHACIt));
    }

    bool operator!=(const IteratorT<IsConst> &Rhs) const {
      return &MHost != &Rhs.MHost || MGenericIsActive != Rhs.MGenericIsActive ||
             ((MGenericIsActive && MGCIt != Rhs.MGCIt) ||
              (!MGenericIsActive && MHACIt != Rhs.MHACIt));
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

      if (!MGenericIsActive && MHACIt != MHost.endHostAccessor())
        return *MHACIt;

      assert(false);

      return nullptr;
    }

  private:
    void increment() {
      if (MGenericIsActive) {
        ++MGCIt;

        if (MGCIt == MHost.MGenericCommands.end()) {
          MGenericIsActive = false;
          MHACIt = MHost.MHostAccessorCommands.begin();
          return;
        }

        return;
      }

      assert(MGCIt == MHost.MGenericCommands.end());

      if (MHACIt == MHost.endHostAccessor())
        return;

      ++MHACIt;
    }
  };
};

} // namespace detail
} // namespace sycl
} // __SYCL_INLINE_NAMESPACE(cl)
