// REQUIRES: cpu
// RUN: %{build} -Wno-error=deprecated-declarations -fsycl-id-queries-range=uint -o %t.out
// RUN: %{run} %t.out

#include <climits>
#include <sycl/detail/core.hpp>

namespace S = sycl;

constexpr char Msg[] =
    "Provided range and/or offset does not fit in unsigned int. "
    "Pass `-fsycl-id-queries-range=none' to "
    "remove this limit.";

void checkRangeException(S::exception &E) {
  std::cerr << E.what() << std::endl;

  assert(std::string(E.what()).find(Msg) == 0 && "Unexpected message");
}

void checkOffsetException(S::exception &E) {
  std::cerr << E.what() << std::endl;

  assert(std::string(E.what()).find(Msg) == 0 && "Unexpected message");
}

int main(void) {
  auto EH = [](S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  if constexpr (sizeof(size_t) <= sizeof(unsigned int))
    return 0;

  static constexpr size_t OutOfLimitsSize = static_cast<size_t>(UINT_MAX) + 1;
  static constexpr size_t InLimitsSize = static_cast<size_t>(UINT_MAX);

  S::range<2> RangeOutOfLimits{OutOfLimitsSize, 1};
  S::range<2> RangeInLimits{1, 1};
  S::range<2> RangeInLimits_Large{InLimitsSize / 2, 1};
  S::id<2> OffsetOutOfLimits{OutOfLimitsSize, 1};
  S::id<2> OffsetInLimits{1, 1};
  S::id<2> OffsetInLimits_Large{(InLimitsSize / 4) * 3, 1};

  int Data = 0;
  S::buffer<int, 1> Buf{&Data, 1};

  // Test 1: no offset, either dim of range exceeds UINT_MAX
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ROL_UINT>(RangeOutOfLimits,
                                          [=](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::exception &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // Test 2: no offset, all dims of range are in UINT_MAX limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_UINT>(RangeInLimits,
                                          [Acc](S::id<2> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // Test 3: large offset exceeds UINT_MAX
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OOL_UINT>(
          RangeInLimits, OffsetOutOfLimits,
          [Acc](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::exception &E) {
    checkOffsetException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // Test 4: small offset, neither range dim exceeds UINT_MAX
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OIL_UINT>(
          RangeInLimits, OffsetInLimits, [Acc](S::id<2> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // Test 5: range + offset exceeds UINT_MAX
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OIL_SOL_UINT>(
          RangeInLimits_Large, OffsetInLimits_Large,
          [Acc](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::exception &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // Test 6: Values just below INT_MAX but above INT_MAX should succeed with
  // UINT mode This proves UINT mode allows larger values than INT mode
  static constexpr size_t JustAboveIntMax = static_cast<size_t>(INT_MAX) + 1024;
  S::range<1> RangeJustAboveIntMax{JustAboveIntMax};

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ABOVE_INT_MAX>(RangeJustAboveIntMax,
                                               [Acc](S::id<1> Id) {
                                                 // This kernel would fail in
                                                 // INT mode but succeeds in
                                                 // UINT mode
                                                 if (Id[0] == 0)
                                                   Acc[0] += 1;
                                               });
    });
  } catch (...) {
    assert(false &&
           "Unexpected exception: UINT mode should allow values > INT_MAX");
  }
  return 0;
}
