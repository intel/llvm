// RUN: %clangxx -fsycl -fsycl-id-queries-fit-in-int -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out

#include <climits>
#include <iostream>
#include <sycl/sycl.hpp>

namespace S = sycl;

void checkRangeException(S::runtime_error &E) {
  constexpr char Msg[] = "Provided range is out of integer limits. "
                         "Pass `-fno-sycl-id-queries-fit-in-int' to "
                         "disable range check.";

  std::cerr << E.what() << std::endl;

  assert(std::string(E.what()).find(Msg) == 0 && "Unexpected message");
}

void checkOffsetException(S::runtime_error &E) {
  constexpr char Msg[] = "Provided offset is out of integer limits. "
                         "Pass `-fno-sycl-id-queries-fit-in-int' to "
                         "disable offset check.";

  std::cerr << E.what() << std::endl;

  assert(std::string(E.what()).find(Msg) == 0 && "Unexpected message");
}

void test() {
  auto EH = [](S::exception_list EL) {
    for (const std::exception_ptr &E : EL) {
      throw E;
    }
  };

  S::queue Queue(EH);

  static constexpr size_t OutOfLimitsSize = static_cast<size_t>(INT_MAX) + 1;

  S::range<2> RangeOutOfLimits{OutOfLimitsSize, 1};
  S::range<2> RangeInLimits{1, 1};
  S::range<2> RangeInLimits_POL{OutOfLimitsSize / 2, 3};
  S::range<2> RangeInLimits_Large{OutOfLimitsSize / 2, 1};
  S::id<2> OffsetOutOfLimits{OutOfLimitsSize, 1};
  S::id<2> OffsetInLimits{1, 1};
  S::id<2> OffsetInLimits_Large{(OutOfLimitsSize / 4) * 3, 1};
  S::nd_range<2> NDRange_ROL_LIL_OIL{RangeOutOfLimits, RangeInLimits,
                                     OffsetInLimits};
  S::nd_range<2> NDRange_RIL_LOL_OIL{RangeInLimits, RangeOutOfLimits,
                                     OffsetInLimits};
  S::nd_range<2> NDRange_RIL_LIL_OOL{RangeInLimits, RangeInLimits,
                                     OffsetOutOfLimits};
  S::nd_range<2> NDRange_RIL_LIL_OIL(RangeInLimits, RangeInLimits,
                                     OffsetInLimits);
  S::nd_range<2> NDRange_RIL_LIL_OIL_POL(S::range<2>{OutOfLimitsSize / 2, 3},
                                         S::range<2>{OutOfLimitsSize / 2, 1});
  S::nd_range<2> NDRange_RIL_LIL_OIL_SOL(
      S::range<2>{OutOfLimitsSize / 2, 1}, S::range<2>{OutOfLimitsSize / 2, 1},
      S::id<2>{(OutOfLimitsSize / 4) * 3, (OutOfLimitsSize / 4) * 3});

  int Data = 0;
  S::buffer<int, 1> Buf{&Data, 1};

  // no offset, either dim of range exceeds limit
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ROL>(RangeOutOfLimits,
                                     [=](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // no offset, all dims of range are in limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL>(RangeInLimits,
                                     [Acc](S::id<2> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // no offset, all dims of range are in limits, linear id exceeds limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_POL>(RangeInLimits_POL,
                                         [Acc](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // small offset, either dim of range exceeds limit
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ROL_OIL>(RangeOutOfLimits, OffsetInLimits,
                                         [Acc](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // large offset, neither dim of range exceeds limit, offset + range > limit
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OIL_SOL>(
          RangeInLimits_Large, OffsetInLimits_Large,
          [Acc](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // large offset, neither dim of range exceeds limit
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OOL>(RangeInLimits, OffsetOutOfLimits,
                                         [Acc](S::id<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkOffsetException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // small offset, neither range dim exceeds limit
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OIL>(RangeInLimits, OffsetInLimits,
                                         [Acc](S::id<2> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // small offset, global range's dim is out of limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GOL_LIL_OIL>(
          NDRange_ROL_LIL_OIL, [Acc](S::nd_item<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // small offset, local range is out of limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LOL_OIL>(
          NDRange_RIL_LOL_OIL, [Acc](S::nd_item<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // large offset, ranges are in limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LIL_OOL>(
          NDRange_RIL_LIL_OOL, [Acc](S::nd_item<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkOffsetException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // small offset, ranges are in limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LIL_OIL>(
          NDRange_RIL_LIL_OIL, [Acc](S::nd_item<2> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // small offset, ranges are in limits, linear id out of limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LIL_OIL_POL>(
          NDRange_RIL_LIL_OIL_POL, [Acc](S::nd_item<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  // small offset, ranges are in limits, range + offset exceeds limits
  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LIL_OIL_SOL>(
          NDRange_RIL_LIL_OIL_POL, [Acc](S::nd_item<2> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }
}

int main(void) {
  test();
  return 0;
}
