// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple -D__SYCL_ID_QUERIES_FIT_IN_INT__=1 %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <climits>
#include <CL/sycl.hpp>

namespace S = cl::sycl;

void checkRangeException(S::runtime_error &E) {
  constexpr char Msg[] = "Provided range is out of integer limits. Suggest "
                         "disabling `id-queries-fit-in-int32' optimizations "
                         "flag.";

  std::cerr << E.what() << std::endl;

  assert(std::string(E.what()).find(Msg) == 0 && "Unexpected message");
}

void checkOffsetException(S::runtime_error &E) {
  constexpr char Msg[] = "Provided offset is out of integer limits. Suggest "
                         "disabling `id-queries-fit-in-int32' optimizations "
                         "flag.";

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

  S::range<1> RangeOutOfLimits{OutOfLimitsSize};
  S::range<1> RangeInLimits{1};
  S::id<1> OffsetOutOfLimits{OutOfLimitsSize};
  S::id<1> OffsetInLimits{1};
  S::nd_range<1> NDRangeROLLILOIL{RangeOutOfLimits, RangeInLimits,
                                  OffsetInLimits};
  S::nd_range<1> NDRangeRILLOLOIL{RangeInLimits, RangeOutOfLimits,
                                  OffsetInLimits};
  S::nd_range<1> NDRangeRILLILOOL{RangeInLimits, RangeInLimits,
                                  OffsetOutOfLimits};
  S::nd_range<1> NDRangeRILLILOIL(RangeInLimits, RangeInLimits, OffsetInLimits);

  int Data = 0;
  S::buffer<int, 1> Buf{&Data, 1};

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ROL>(RangeOutOfLimits,
                                     [=](S::id<1> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL>(RangeInLimits,
                                     [Acc](S::id<1> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ROL_OIL>(RangeOutOfLimits, OffsetInLimits,
                                         [Acc](S::id<1> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OOL>(RangeInLimits, OffsetOutOfLimits,
                                         [Acc](S::id<1> Id) { Acc[0] += 1; });
    });

    assert(false && "Exception expected");
  } catch (S::runtime_error &E) {
    checkOffsetException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_RIL_OIL>(RangeInLimits, OffsetInLimits,
                                         [Acc](S::id<1> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GOL_LIL_OIL>(
          NDRangeROLLILOIL, [Acc](S::nd_item<1> Id) { Acc[0] += 1; });
    });
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LOL_OIL>(
          NDRangeRILLOLOIL, [Acc](S::nd_item<1> Id) { Acc[0] += 1; });
    });
  } catch (S::runtime_error &E) {
    checkRangeException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LIL_OOL>(
          NDRangeRILLILOOL, [Acc](S::nd_item<1> Id) { Acc[0] += 1; });
    });
  } catch (S::runtime_error &E) {
    checkOffsetException(E);
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }

  try {
    Queue.submit([&](S::handler &CGH) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(CGH);

      CGH.parallel_for<class PF_ND_GIL_LIL_OIL>(
          NDRangeRILLILOIL, [Acc](S::nd_item<1> Id) { Acc[0] += 1; });
    });
  } catch (...) {
    assert(false && "Unexpected exception catched");
  }
}

int main(void) {
  test();
  return 0;
}
