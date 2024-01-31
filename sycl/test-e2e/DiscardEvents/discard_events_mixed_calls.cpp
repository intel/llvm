// RUN: %{build} -o %t.out

// The purpose of all tests is to make sure in-order semantics works correctly
// using discard_events and alternating event and eventless kernel calls in
// different ways.

// The test checks that eventless kernel calls work correctly after several
// event kernel calls.
// RUN: %{run} %t.out accessor-usm

// The test checks that event kernel calls work correctly after several
// eventless kernel calls.
// RUN: %{run} %t.out usm-accessor

// The test checks that alternating event and eventless kernel calls work
// correctly.
// RUN: %{run} %t.out mixed

// The test checks that piEnqueueMemBufferMap and piEnqueueMemUnmap work
// correctly when we alternate between event and eventless kernel calls.
// RUN: %{run} %t.out map-unmap

// Note that the tests use buffer functionality and if you have problems with
// the tests, please check if they pass without the discard_events property, if
// they don't pass then it's most likely a general issue unrelated to
// discard_events.

#include <CL/sycl.hpp>
#include <cassert>
#include <iostream>

using namespace cl::sycl;
static constexpr size_t BUFFER_SIZE = 1024;
static constexpr int MAX_ITER_NUM1 = 10;
static constexpr int MAX_ITER_NUM2 = 10;

void TestHelper(sycl::queue Q,
                const std::function<void(sycl::range<1> Range, int *Harray,
                                         sycl::buffer<int, 1> Buf)> &Function) {
  if (!Q.get_device().has(aspect::usm_shared_allocation))
    return;
  int *Harray = sycl::malloc_shared<int>(BUFFER_SIZE, Q);
  assert(Harray != nullptr);
  for (size_t i = 0; i < BUFFER_SIZE; ++i) {
    Harray[i] = 0;
  }

  sycl::range<1> Range(BUFFER_SIZE);
  sycl::buffer<int, 1> Buf(Range);

  Function(Range, Harray, Buf);

  free(Harray, Q);
}

void IfTrueIncrementUSM(sycl::queue Q, sycl::range<1> Range, int *Harray,
                        int ValueToCheck) {
  Q.submit([&](sycl::handler &CGH) {
    CGH.parallel_for<class increment_usm>(Range, [=](sycl::item<1> itemID) {
      size_t i = itemID.get_id(0);
      if (Harray[i] == ValueToCheck) {
        Harray[i] += 1;
      }
    });
  });
}

void IfTrueIncrementBufferAndUSM(sycl::queue Q, sycl::range<1> Range,
                                 int *Harray, sycl::buffer<int, 1> Buf,
                                 int ValueToCheck) {
  Q.submit([&](sycl::handler &CGH) {
    auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
    CGH.parallel_for<class increment_buffer_and_usm>(
        Range, [=](sycl::item<1> itemID) {
          size_t i = itemID.get_id(0);
          if (Harray[i] == ValueToCheck) {
            ++Acc[i];
            ++Harray[i];
          }
        });
  });
}

void RunTest_USM_Accessor(sycl::queue Q) {
  TestHelper(Q, [&](sycl::range<1> Range, int *Harray,
                    sycl::buffer<int, 1> Buf) {
    {
      sycl::host_accessor HostAcc(Buf);
      for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        HostAcc[i] = 0;
      }
    }

    for (int i = 0; i < MAX_ITER_NUM1; ++i)
      IfTrueIncrementUSM(Q, Range, Harray, (i));

    for (int i = 0; i < MAX_ITER_NUM2; ++i)
      IfTrueIncrementBufferAndUSM(Q, Range, Harray, Buf, (MAX_ITER_NUM1 + i));

    Q.wait();

    // check results
    for (size_t i = 0; i < BUFFER_SIZE; ++i) {
      int expected = MAX_ITER_NUM1 + MAX_ITER_NUM2;
      assert(Harray[i] == expected);
    }
    {
      sycl::host_accessor HostAcc(Buf, sycl::read_only);
      for (size_t i = 0; i < BUFFER_SIZE; ++i) {
        int expected = MAX_ITER_NUM2;
        assert(HostAcc[i] == expected);
      }
    }
  });
}

void RunTest_Accessor_USM(sycl::queue Q) {
  TestHelper(Q,
             [&](sycl::range<1> Range, int *Harray, sycl::buffer<int, 1> Buf) {
               {
                 sycl::host_accessor HostAcc(Buf);
                 for (size_t i = 0; i < BUFFER_SIZE; ++i) {
                   HostAcc[i] = 0;
                 }
               }

               for (int i = 0; i < MAX_ITER_NUM1; ++i)
                 IfTrueIncrementBufferAndUSM(Q, Range, Harray, Buf, (i));

               for (int i = 0; i < MAX_ITER_NUM2; ++i)
                 IfTrueIncrementUSM(Q, Range, Harray, (MAX_ITER_NUM1 + i));

               Q.wait();

               // check results
               for (size_t i = 0; i < BUFFER_SIZE; ++i) {
                 int expected = MAX_ITER_NUM1 + MAX_ITER_NUM2;
                 assert(Harray[i] == expected);
               }
               {
                 sycl::host_accessor HostAcc(Buf, sycl::read_only);
                 for (size_t i = 0; i < BUFFER_SIZE; ++i) {
                   int expected = MAX_ITER_NUM1;
                   assert(HostAcc[i] == expected);
                 }
               }
             });
}

void RunTest_Mixed(sycl::queue Q) {
  TestHelper(
      Q, [&](sycl::range<1> Range, int *Harray, sycl::buffer<int, 1> Buf) {
        {
          sycl::host_accessor HostAcc(Buf);
          for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            HostAcc[i] = 0;
          }
        }

        for (int i = 0; i < MAX_ITER_NUM1; ++i) {
          IfTrueIncrementUSM(Q, Range, Harray, (i * 2));
          IfTrueIncrementBufferAndUSM(Q, Range, Harray, Buf, (i * 2 + 1));
        }

        for (int i = 0; i < MAX_ITER_NUM2; ++i) {
          IfTrueIncrementBufferAndUSM(Q, Range, Harray, Buf,
                                      (MAX_ITER_NUM1 * 2 + i * 2));
          IfTrueIncrementUSM(Q, Range, Harray, (MAX_ITER_NUM1 * 2 + i * 2 + 1));
        }

        Q.wait();

        // check results
        for (size_t i = 0; i < BUFFER_SIZE; ++i) {
          int expected = MAX_ITER_NUM1 * 2 + MAX_ITER_NUM2 * 2;
          assert(Harray[i] == expected);
        }
        {
          sycl::host_accessor HostAcc(Buf, sycl::read_only);
          for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            int expected = MAX_ITER_NUM1 + MAX_ITER_NUM2;
            assert(HostAcc[i] == expected);
          }
        }
      });
}

void RunTest_MemBufferMapUnMap(sycl::queue Q) {
  TestHelper(
      Q, [&](sycl::range<1> Range, int *Harray, sycl::buffer<int, 1> Buf) {
        Q.submit([&](sycl::handler &CGH) {
          auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
          CGH.parallel_for<class kernel1>(Range, [=](sycl::item<1> itemID) {
            size_t i = itemID.get_id(0);
            Harray[i] = i;
            Acc[i] = i;
          });
        });

        Q.submit([&](sycl::handler &CGH) {
          CGH.parallel_for<class kernel2>(Range, [=](sycl::item<1> itemID) {
            size_t i = itemID.get_id(0);
            if (Harray[i] == i)
              Harray[i] += 10;
          });
        });

        {
          // waiting for all queue operations in piEnqueueMemBufferMap and then
          // checking buffer
          sycl::host_accessor HostAcc(Buf);
          for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            int expected = i;
            assert(HostAcc[i] == expected);
          }
          for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            HostAcc[i] += 10;
          }
        }

        Q.submit([&](sycl::handler &CGH) {
          CGH.parallel_for<class kernel3>(Range, [=](sycl::item<1> itemID) {
            size_t i = itemID.get_id(0);
            if (Harray[i] == (i + 10))
              Harray[i] += 100;
          });
        });

        Q.submit([&](sycl::handler &CGH) {
          // waiting for all queue operations in piEnqueueMemUnmap and then
          // using buffer
          auto Acc = Buf.get_access<sycl::access::mode::read_write>(CGH);
          CGH.parallel_for<class kernel4>(Range, [=](sycl::item<1> itemID) {
            size_t i = itemID.get_id(0);
            if (Acc[i] == (i + 10))
              if (Harray[i] == (i + 110)) {
                Harray[i] += 1000;
                Acc[i] += 100;
              }
          });
        });
        Q.wait();

        // check results
        for (size_t i = 0; i < BUFFER_SIZE; ++i) {
          int expected = i + 1110;
          assert(Harray[i] == expected);
        }
        {
          sycl::host_accessor HostAcc(Buf, sycl::read_only);
          for (size_t i = 0; i < BUFFER_SIZE; ++i) {
            int expected = i + 110;
            assert(HostAcc[i] == expected);
          }
        }
      });
}

int main(int Argc, const char *Argv[]) {
  assert(Argc == 2 && "Invalid number of arguments");
  std::string TestType(Argv[1]);

  sycl::property_list props{
      sycl::property::queue::in_order{},
      sycl::ext::oneapi::property::queue::discard_events{}};
  sycl::queue Q(props);

  if (TestType == "accessor-usm") {
    std::cerr << "RunTest_Accessor_USM" << std::endl;
    RunTest_Accessor_USM(Q);
  } else if (TestType == "usm-accessor") {
    std::cerr << "RunTest_USM_Accessor" << std::endl;
    RunTest_USM_Accessor(Q);
  } else if (TestType == "mixed") {
    std::cerr << "RunTest_Mixed" << std::endl;
    RunTest_Mixed(Q);
  } else if (TestType == "map-unmap") {
    std::cerr << "RunTest_MemBufferMapUnMap" << std::endl;
    RunTest_MemBufferMapUnMap(Q);
  } else {
    assert(0 && "Unsupported test type!");
  }

  std::cout << "The test passed." << std::endl;
  return 0;
}
