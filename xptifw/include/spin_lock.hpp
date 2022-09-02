//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
#pragma once

#include <atomic>
#include <thread>

#ifndef __has_include
#define __has_include(x)
#endif

#if __has_include(<immintrin.h>)
#define HAS_PAUSE
#include <immintrin.h>
#endif

namespace xpti {

namespace detail {
class Backoff {
public:
  void pause() {
    if (MCount <= LOOPS_BEFORE_YIELD) {
#ifdef HAS_PAUSE
      _mm_pause();
#else
      std::this_thread::yield();
#endif
      MCount *= 2;
    } else {
      std::this_thread::yield();
    }
  }

  void reset() noexcept { MCount = 1; }

private:
  static constexpr uint32_t LOOPS_BEFORE_YIELD = 16;
  uint32_t MCount = 1;
};
} // namespace detail

/// RAII-style read lock for \c SharedSpinLock.
///
/// Unlike std::shared_lock this class provides aims to upgrade reader lock to
/// writer lock, which in some cases can improve performance.
template <typename Mutex> class SharedLock {
public:
  SharedLock() = default;

  SharedLock(Mutex &M) { acquire(M); }

  void acquire(Mutex &M) {
    MMutex = &M;
    M.lock_shared();
  }

  void release() {
    if (MIsWriter)
      MMutex->unlock();
    else
      MMutex->unlock_shared();
    MMutex = nullptr;
    MIsWriter = false;
  }

  void upgrade_to_writer() {
    if (!MIsWriter) {
      MIsWriter = true;
      MMutex->upgrade();
    }
  }

  ~SharedLock() {
    if (MMutex)
      release();
  }

private:
  Mutex *MMutex;
  bool MIsWriter = false;
};

/// SpinLock is a synchronization primitive, that uses atomic variable and
/// causes thread trying acquire lock wait in loop while repeatedly check if
/// the lock is available.
///
/// One important feature of this implementation is that std::atomic_flag can
/// be zero-initialized. This allows SpinLock to have trivial constructor and
/// destructor, which makes it possible to use it in global context (unlike
/// std::mutex, that doesn't provide such guarantees).
class SpinLock {
public:
  void lock() {
    detail::Backoff B;
    while (MLock.test_and_set(std::memory_order_acquire))
      B.pause();
  }
  void unlock() { MLock.clear(std::memory_order_release); }

private:
  std::atomic_flag MLock = ATOMIC_FLAG_INIT;
};

/// SharedSpinLock is a synchronization primitive, that allows RW-locks.
///
/// Unlike std::shared_mutex, SharedSpinLock is guaranteed to be trivially
/// destructible. It also provides aims to upgrade reader lock to writer lock.
class SharedSpinLock {
public:
  void lock() noexcept {
    for (detail::Backoff B;; B.pause()) {
      uint32_t CurState = MState.load(std::memory_order_relaxed);
      if (!(CurState & BUSY)) {
        if (MState.compare_exchange_strong(CurState, WRITER)) {
          break;
        }
        B.reset();
      } else if (!(CurState & WRITER_PENDING)) {
        MState |= WRITER_PENDING;
      }
    }
  }

  void unlock() noexcept { MState &= READERS; }

  void lock_shared() noexcept {
    for (detail::Backoff B;; B.pause()) {
      uint32_t CurState = MState.load(std::memory_order_relaxed);
      if (!(CurState & (WRITER | WRITER_PENDING))) {
        uint32_t OldState = MState.fetch_add(ONE_READER);
        if (!(OldState & WRITER))
          break;

        MState -= ONE_READER;
      }
    }
  }

  void unlock_shared() noexcept {
    MState.fetch_sub(ONE_READER, std::memory_order_release);
  }

  void upgrade() noexcept {
    uint32_t CurState = MState.load(std::memory_order_relaxed);
    if ((CurState & READERS) == ONE_READER || !(CurState & WRITER_PENDING)) {
      if (MState.compare_exchange_strong(CurState,
                                         CurState | WRITER | WRITER_PENDING)) {
        detail::Backoff B;
        while ((MState.load(std::memory_order_relaxed) & READERS) !=
               ONE_READER) {
          B.pause();
        }

        MState -= (ONE_READER + WRITER_PENDING);
        return;
      }
    }
    unlock_shared();
    lock();
  }

private:
  static constexpr uint32_t WRITER = 1 << 31;
  static constexpr uint32_t WRITER_PENDING = 1 << 30;
  static constexpr uint32_t READERS = ~(WRITER | WRITER_PENDING);
  static constexpr uint32_t ONE_READER = 1;
  static constexpr uint32_t BUSY = WRITER | READERS;
  std::atomic<uint32_t> MState{0};
};
} // namespace xpti
