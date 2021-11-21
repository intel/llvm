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

  void reset() noexcept { MCount = 0; }

private:
  static constexpr uint32_t LOOPS_BEFORE_YIELD = 16;
  uint32_t MCount = 0;
};
} // namespace detail

/// This is an implementation of a SpinLock synchronization primitive, that has
/// trivial constructor and destructor.
class SpinLock {
public:
  void lock() noexcept {
    for (detail::Backoff B; MLock.exchange(true, std::memory_order_acquire);
         B.pause())
      ;
  }
  void unlock() noexcept { MLock.store(false, std::memory_order_release); }

  bool is_locked() const noexcept { return MLock == true; }

private:
  std::atomic<bool> MLock{false};
};

class SharedSpinLock {
public:
  void lock() noexcept {
    for (detail::Backoff B;; B.pause()) {
      uint32_t CurState = MState.load(std::memory_order_relaxed);
      if (CurState == 0) {
        if (MState.compare_exchange_strong(CurState, WRITER)) {
          break;
        }
        B.reset();
      }
    }
  }

  void unlock() noexcept { MState &= ~WRITER; }

  void lock_shared() noexcept {
    for (detail::Backoff B;; B.pause()) {
      uint32_t CurState = MState.load(std::memory_order_relaxed);
      if (!(CurState & WRITER)) {
        uint32_t OldState = MState.fetch_add(READER);
        if (!(OldState & WRITER))
          break;

        MState -= READER;
      }
    }
  }

  void unlock_shared() noexcept {
    MState.fetch_sub(READER, std::memory_order_release);
  }

private:
  static constexpr uint32_t WRITER = 1 << 31;
  static constexpr uint32_t READER = 1;
  std::atomic<uint32_t> MState{0};
};
} // namespace xpti
