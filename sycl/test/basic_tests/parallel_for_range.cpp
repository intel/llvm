// RUN: %clangxx -fsycl %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <CL/sycl.hpp>

#include <iostream>

using namespace cl::sycl;

[[cl::reqd_work_group_size(4, 4, 4)]] void reqd_wg_size_helper() {
  // do nothing
}

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);
  device D(Q.get_device());

  string_class DeviceVendorName = D.get_info<info::device::vendor>();
  auto DeviceType = D.get_info<info::device::device_type>();

  // parallel_for, (16, 16, 16) global, (8, 8, 8) local, reqd_wg_size(4, 4, 4)
  // -> fail
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class ReqdWGSizeNegativeA>(
          nd_range<3>(range<3>(16, 16, 16), range<3>(8, 8, 8)),
          [=](nd_item<3>) { reqd_wg_size_helper(); });
    });
    Q.wait_and_throw();
    std::cerr << "Test case ReqdWGSizeNegativeA failed: no exception has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (nd_range_error &E) {
    if (string_class(E.what()).find(
            "Specified local size doesn't match the required work-group size "
            "specified in the program source") == string_class::npos) {
      std::cerr
          << "Test case ReqdWGSizeNegativeA failed: unexpected exception: "
          << E.what() << std::endl;
      return 1;
    }
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizeNegativeA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  string_class OCLVersionStr = D.get_info<info::device::version>();
  assert(OCLVersionStr.size() >= 10 &&
         "Unexpected device version string"); // strlen("OpenCL X.Y")
  const char *OCLVersion = &OCLVersionStr[7]; // strlen("OpenCL ")
  if (OCLVersion[0] == '1' || (OCLVersion[0] == '2' && OCLVersion[2] == '0')) {
    // parallel_for, (16, 16, 16) global, null local, reqd_wg_size(4, 4, 4) //
    // -> fail
    try {
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<class ReqdWGSizeNegativeB>(
            range<3>(16, 16, 16), [=](item<3>) { reqd_wg_size_helper(); });
      });
      Q.wait_and_throw();
      std::cerr
          << "Test case ReqdWGSizeNegativeB failed: no exception has been "
             "thrown\n";
      return 1; // We shouldn't be here, exception is expected
    } catch (nd_range_error &E) {
      if (string_class(E.what()).find(
              "OpenCL 1.x and 2.0 requires to pass local size argument even if "
              "required work-group size was specified in the program source") ==
          string_class::npos) {
        std::cerr
            << "Test case ReqdWGSizeNegativeB failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      }
    } catch (runtime_error &E) {
      std::cerr
          << "Test case ReqdWGSizeNegativeB failed: unexpected exception: "
          << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case ReqdWGSizeNegativeB failed: something unexpected "
                   "has been caught"
                << std::endl;
      return 1;
    }
  }

  // Positive test-cases that should pass on any underlying OpenCL runtime

  // parallel_for, (8, 8, 8) global, (4, 4, 4) local, reqd_wg_size(4, 4, 4) ->
  // pass
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class ReqdWGSizePositiveA>(
          nd_range<3>(range<3>(8, 8, 8), range<3>(4, 4, 4)),
          [=](nd_item<3>) { reqd_wg_size_helper(); });
    });
    Q.wait_and_throw();
  } catch (nd_range_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  } catch (runtime_error &E) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: unexpected exception: "
              << E.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Test case ReqdWGSizePositiveA failed: something unexpected "
                 "has been caught"
              << std::endl;
    return 1;
  }

  if (OCLVersion[0] == '1') {
    // OpenCL 1.x

    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and
    // number of workitems specified by global_work_size is not evenly
    // divisible by size of work-group given by local_work_size
    try {
      // parallel_for, 100 global, 3 local -> fail
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<class OpenCL1XNegativeA>(
            nd_range<1>(range<1>(100), range<1>(3)), [=](nd_item<1>) {});
      });
      Q.wait_and_throw();
      // FIXME: some Intel runtimes contain bug and don't return expected
      // error code
      if (info::device_type::accelerator != DeviceType ||
          DeviceVendorName.find("Intel") == string_class::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeA failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      }
    } catch (nd_range_error &E) {
      if (string_class(E.what()).find("Non-uniform work-groups are not "
                                      "supported by the target device") ==
          string_class::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeA failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      }
    } catch (runtime_error &E) {
      std::cerr << "Test case OpenCL1XNegativeA failed: unexpected exception: "
                << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case OpenCL1XNegativeA failed: something unexpected "
                   "has been caught"
                << std::endl;
      return 1;
    }

    try {
      // parallel_for, (100, 33, 16) global, (2, 3, 5) local -> fail
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<class OpenCL1XNegativeB>(
            nd_range<3>(range<3>(100, 33, 16), range<3>(2, 3, 5)),
            [=](nd_item<3>) {});
      });
      Q.wait_and_throw();
      // FIXME: some Intel runtimes contain bug and don't return expected
      // error code
      if (info::device_type::accelerator != DeviceType ||
          DeviceVendorName.find("Intel") == string_class::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeB failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      }
    } catch (nd_range_error &E) {
      if (string_class(E.what()).find("Non-uniform work-groups are not "
                                      "supported by the target device") ==
          string_class::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeB failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      }
    } catch (runtime_error &E) {
      std::cerr << "Test case OpenCL1XNegativeB failed: unexpected exception: "
                << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case OpenCL1XNegativeB failed: something unexpected "
                   "has been caught"
                << std::endl;
      return 1;
    }

    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the
    // total number of work-items in the work-group computed as
    // local_work_size[0] * ... * local_work_size[work_dim – 1] is greater
    // than the value specified by CL_DEVICE_MAX_WORK_GROUP_SIZE in
    // table 4.3
    size_t MaxDeviceWGSize = D.get_info<info::device::max_work_group_size>();
    try {
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<class OpenCL1XNegativeC>(
            nd_range<2>(range<2>(MaxDeviceWGSize, MaxDeviceWGSize),
                        range<2>(MaxDeviceWGSize, 2)),
            [=](nd_item<2>) {});
      });
      Q.wait_and_throw();
      std::cerr << "Test case OpenCL1XNegativeC failed: no exception has been "
                   "thrown\n";
      return 1; // We shouldn't be here, exception is expected
    } catch (nd_range_error &E) {
      if (string_class(E.what()).find(
              "Total number of work-items in a work-group cannot exceed "
              "info::device::max_work_group_size which is equal to " +
              std::to_string(MaxDeviceWGSize)) == string_class::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeC failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      }
    } catch (runtime_error &E) {
      std::cerr << "Test case OpenCL1XNegativeC failed: unexpected exception: "
                << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case OpenCL1XNegativeC failed: something unexpected "
                   "has been caught"
                << std::endl;
      return 1;
    }
  } else if (OCLVersion[0] == '2') {
    // OpenCL 2.x

    // OpenCL 2.x:
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and the
    // total number of work-items in the work-group computed as
    // local_work_size[0] * ... * local_work_size[work_dim – 1] is greater
    // than the value specified by CL_KERNEL_WORK_GROUP_SIZE in table 5.21.
    {
      program P(Q.get_context());
      P.build_with_kernel_type<class OpenCL2XNegativeA>();

      kernel K = P.get_kernel<class OpenCL2XNegativeA>();
      size_t MaxKernelWGSize =
          K.get_work_group_info<info::kernel_work_group::work_group_size>(
              Q.get_device());
      try {
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeA>(
              K,
              nd_range<2>(range<2>(MaxKernelWGSize, MaxKernelWGSize),
                          range<2>(MaxKernelWGSize, 2)),
              [=](nd_item<2>) {});
        });
        Q.wait_and_throw();
        std::cerr
            << "Test case OpenCL2XNegativeA failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      } catch (nd_range_error &E) {
        if (string_class(E.what()).find(
                "Total number of work-items in a work-group cannot exceed "
                "info::kernel_work_group::work_group_size which is equal to " +
                std::to_string(MaxKernelWGSize) + " for this kernel") ==
            string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeA failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeA failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "Test case OpenCL2XNegativeA failed: something unexpected "
                     "has been caught"
                  << std::endl;
        return 1;
      }
    }

    // By default, program is built in OpenCL 1.2 mode, so the following error
    // is expected even for OpenCL 2.x:
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified and
    // number of workitems specified by global_work_size is not evenly
    // divisible by size of work-group given by local_work_size
    {
      try {
        // parallel_for, 100 global, 3 local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeB>(
              nd_range<1>(range<1>(100), range<1>(3)), [=](nd_item<1>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeB failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (string_class(E.what()).find(
                "Non-uniform work-groups are not allowed by default. "
                "Underlying OpenCL 2.x implementation supports this feature "
                "and to enable it, build device program with -cl-std=CL2.0") ==
            string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeB failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeB failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "Test case OpenCL2XNegativeB failed: something unexpected "
                     "has been caught"
                  << std::endl;
        return 1;
      }

      try {
        // parallel_for, (100, 33, 16) global, (2, 3, 5) local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeC>(
              nd_range<3>(range<3>(100, 33, 16), range<3>(2, 3, 5)),
              [=](nd_item<3>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeC failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (string_class(E.what()).find(
                "Non-uniform work-groups are not allowed by default. "
                "Underlying OpenCL 2.x implementation supports this feature "
                "and to enable it, build device program with -cl-std=CL2.0") ==
            string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeC failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeC failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "Test case OpenCL2XNegativeC failed: something unexpected "
                     "has been caught"
                  << std::endl;
        return 1;
      }
    }

    // Enable non-uniform work-groups by passing -cl-std=CL2.0
    {
      program P(Q.get_context());
      P.build_with_kernel_type<class OpenCL2XPositiveA>("-cl-std=CL2.0");

      kernel K = P.get_kernel<class OpenCL2XPositiveA>();
      try {
        // parallel_for, 100 global, 3 local -> pass
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XPositiveA>(
              K, nd_range<1>(range<1>(100), range<1>(3)), [=](nd_item<1>) {});
        });
        Q.wait_and_throw();
      } catch (nd_range_error &E) {
        std::cerr
            << "Test case OpenCL2XPositiveA failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XPositiveA failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "Test case OpenCL2XPositiveA failed: something unexpected "
                     "has been caught"
                  << std::endl;
        return 1;
      }
    }

    {
      program P(Q.get_context());
      P.build_with_kernel_type<class OpenCL2XPositiveB>("-cl-std=CL2.0");

      kernel K = P.get_kernel<class OpenCL2XPositiveB>();
      try {
        // parallel_for, (100, 33, 16) global, (2, 3, 5) local -> pass
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XPositiveB>(
              K, nd_range<3>(range<3>(100, 33, 16), range<3>(2, 3, 5)),
              [=](nd_item<3>) {});
        });
        Q.wait_and_throw();
      } catch (nd_range_error &E) {
        std::cerr
            << "Test case OpenCL2XPositiveB failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XPositiveB failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "Test case OpenCL2XPositiveB failed: something unexpected "
                     "has been caught"
                  << std::endl;
        return 1;
      }
    }

    // Enable 2.0 mode with non-uniform work-groups, but disable the latter by
    // specifying -cl-uniform-work-group-size:
    // CL_INVALID_WORK_GROUP_SIZE if the program was compiled with
    // –cl-uniform-work-group-size and the number of work-items specified
    // by global_work_size is not evenly divisible by size of work-group
    // given by local_work_size
    {
      program P(Q.get_context());
      P.build_with_kernel_type<class OpenCL2XNegativeD>(
          "-cl-std=CL2.0 -cl-uniform-work-group-size");

      kernel K = P.get_kernel<class OpenCL2XNegativeD>();
      try {
        // parallel_for, 100 global, 3 local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeD>(
              K, nd_range<1>(range<1>(100), range<1>(3)), [=](nd_item<1>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeD failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (string_class(E.what()).find(
                "Non-uniform work-groups are not allowed by default. "
                "Underlying OpenCL 2.x implementation supports this feature, "
                "but it is disabled by -cl-uniform-work-group-size build "
                "flag") == string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeD failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeD failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "Test case OpenCL2XNegativeD failed: something unexpected "
                     "has been caught"
                  << std::endl;
        return 1;
      }
    }

    {
      program P(Q.get_context());
      P.build_with_kernel_type<class OpenCL2XNegativeE>(
          "-cl-std=CL2.0 -cl-uniform-work-group-size");

      kernel K = P.get_kernel<class OpenCL2XNegativeE>();
      try {
        // parallel_for, (100, 33, 16) global, (2, 3, 5) local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeE>(
              K, nd_range<3>(range<3>(100, 33, 16), range<3>(2, 3, 5)),
              [=](nd_item<3>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeE failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (string_class(E.what()).find(
                "Non-uniform work-groups are not allowed by default. "
                "Underlying OpenCL 2.x implementation supports this feature, "
                "but it is disabled by -cl-uniform-work-group-size build "
                "flag") == string_class::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeE failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeE failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr << "Test case OpenCL2XNegativeE failed: something unexpected "
                     "has been caught"
                  << std::endl;
        return 1;
      }
    }
  }

  // local size has a 0-based range -- no SIGFPEs, we hope
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for<class NegativeA>(
          nd_range<2>(range<2>(5, 33), range<2>(1, 0)), [=](nd_item<2>) {});
    });
    Q.wait_and_throw();
    std::cerr << "Test case NegativeA failed: no exception has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (runtime_error) {
  }

  // parallel_for_work_group with 0-based local range
  try {
    Q.submit([&](handler &CGH) {
      CGH.parallel_for_work_group<class NegativeB>(
          range<2>(5, 33), range<2>(1, 0), [=](group<2>) {});
    });
    Q.wait_and_throw();
    std::cerr << "Test case NegativeB failed: no exception has been "
                 "thrown\n";
    return 1; // We shouldn't be here, exception is expected
  } catch (runtime_error) {
  }

  return 0;
}
