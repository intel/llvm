// XFAIL: level_zero&&gpu
// UNSUPPORTED: windows
// Level0 testing times out on Windows.

// RUN: %clangxx -D__SYCL_INTERNAL_API -fsycl -fno-sycl-id-queries-fit-in-int -fsycl-targets=%sycl_triple  %s -o %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

#include <iostream>

using namespace sycl;

int main() {
  auto AsyncHandler = [](exception_list ES) {
    for (auto &E : ES) {
      std::rethrow_exception(E);
    }
  };

  queue Q(AsyncHandler);
  device D(Q.get_device());

  std::string DeviceVendorName = D.get_info<info::device::vendor>();
  auto DeviceType = D.get_info<info::device::device_type>();

  const bool OCLBackend = D.get_platform().get_backend() == backend::opencl;
  std::string OCLVersionStr = D.get_info<info::device::version>();
  assert((OCLVersionStr.size() == 3) && "Unexpected device version string");
  assert(OCLVersionStr.find(".") != std::string::npos &&
         "Unexpected device version string");
  const char OCLVersionMajor = OCLVersionStr[0];
  const char OCLVersionMinor = OCLVersionStr[2];

  // reqd_work_group_size is OpenCL specific.
  if (OCLBackend) {
    if (OCLVersionMajor == '1' ||
        (OCLVersionMajor == '2' && OCLVersionMinor == '0')) {
      // parallel_for, (16, 16, 16) global, (8, 8, 8) local, reqd_wg_size(4, 4,
      // 4)
      // -> fail
      try {
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class ReqdWGSizeNegativeA>(
              nd_range<3>(range<3>(16, 16, 16), range<3>(8, 8, 8)), [=
          ](nd_item<3>) [[sycl::reqd_work_group_size(4, 4, 4)]]{});
        });
        Q.wait_and_throw();
        std::cerr
            << "Test case ReqdWGSizeNegativeA failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find("The specified local size {8, 8, 8} "
                                       "doesn't match the required work-group "
                                       "size specified in the program source "
                                       "{4, 4, 4}") == std::string::npos) {
          std::cerr
              << "Test case ReqdWGSizeNegativeA failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case ReqdWGSizeNegativeA failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr
            << "Test case ReqdWGSizeNegativeA failed: something unexpected "
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
            nd_range<3>(range<3>(8, 8, 8), range<3>(4, 4, 4)), [=
        ](nd_item<3>) [[sycl::reqd_work_group_size(4, 4, 4)]]{});
      });
      Q.wait_and_throw();
    } catch (nd_range_error &E) {
      std::cerr
          << "Test case ReqdWGSizePositiveA failed: unexpected exception: "
          << E.what() << std::endl;
      return 1;
    } catch (runtime_error &E) {
      std::cerr
          << "Test case ReqdWGSizePositiveA failed: unexpected exception: "
          << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case ReqdWGSizePositiveA failed: something unexpected "
                   "has been caught"
                << std::endl;
      return 1;
    }
  } // if  (OCLBackend)

  if (!OCLBackend || (OCLVersionMajor == '1')) {
    // OpenCL 1.x or non-OpenCL backends which behave like OpenCl 1.2 in SYCL.

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
          DeviceVendorName.find("Intel") == std::string::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeA failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      }
    } catch (nd_range_error &E) {
      if (std::string(E.what()).find("Non-uniform work-groups are not "
                                     "supported by the target device") ==
          std::string::npos) {
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
          DeviceVendorName.find("Intel") == std::string::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeB failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      }
    } catch (nd_range_error &E) {
      if (std::string(E.what()).find("Non-uniform work-groups are not "
                                     "supported by the target device") ==
          std::string::npos) {
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

    // Local Size larger than Global.
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified as larger
    // than the global size, then a different error string is used.
    // This is a sub-case of the more general 'non-uniform work group'
    try {
      // parallel_for, 16 global, 17 local -> fail
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<class OpenCL1XNegativeA2>(
            nd_range<1>(range<1>(16), range<1>(17)), [=](nd_item<1>) {});
      });
      Q.wait_and_throw();
      // FIXME: some Intel runtimes contain bug and don't return expected
      // error code
      if (info::device_type::accelerator != DeviceType ||
          DeviceVendorName.find("Intel") == std::string::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeA2 failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      }
    } catch (nd_range_error &E) {
      if ((std::string(E.what()).find("Local workgroup size cannot be greater "
                                      "than global range in any dimension") ==
           std::string::npos) &&
          (std::string(E.what()).find("Non-uniform work-groups are not "
                                      "supported by the target device") ==
           std::string::npos)) {
        std::cerr
            << "Test case OpenCL1XNegativeA2 failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      }
    } catch (runtime_error &E) {
      std::cerr << "Test case OpenCL1XNegativeA2 failed: unexpected exception: "
                << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case OpenCL1XNegativeA2 failed: something unexpected "
                   "has been caught"
                << std::endl;
      return 1;
    }

    // Local Size larger than Global, multi-dimensional
    // This is a sub-case of the more general 'non-uniform work group'
    try {
      // parallel_for, 6, 6, 6 global, 2, 2, 7 local -> fail
      Q.submit([&](handler &CGH) {
        CGH.parallel_for<class OpenCL1XNegativeB2>(
            nd_range<3>(range<3>(6, 6, 6), range<3>(2, 2, 7)),
            [=](nd_item<3>) {});
      });
      Q.wait_and_throw();
      // FIXME: some Intel runtimes contain bug and don't return expected
      // error code
      if (info::device_type::accelerator != DeviceType ||
          DeviceVendorName.find("Intel") == std::string::npos) {
        std::cerr
            << "Test case OpenCL1XNegativeB2 failed: no exception has been "
               "thrown\n";
        return 1; // We shouldn't be here, exception is expected
      }
    } catch (nd_range_error &E) {
      if ((std::string(E.what()).find("Local workgroup size cannot be greater "
                                      "than global range in any dimension") ==
           std::string::npos) &&
          (std::string(E.what()).find("Non-uniform work-groups are not "
                                      "supported by the target device") ==
           std::string::npos)) {
        std::cerr
            << "Test case OpenCL1XNegativeB2 failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      }
    } catch (runtime_error &E) {
      std::cerr << "Test case OpenCL1XNegativeB2 failed: unexpected exception: "
                << E.what() << std::endl;
      return 1;
    } catch (...) {
      std::cerr << "Test case OpenCL1XNegativeB2 failed: something unexpected "
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
      if ((std::string(E.what()).find(
               "Total number of work-items in a work-group cannot exceed " +
               std::to_string(MaxDeviceWGSize)) == std::string::npos) &&
          (std::string(E.what()).find("Non-uniform work-groups are not "
                                      "supported by the target device") ==
           std::string::npos)) {
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
  } else if (OCLBackend && (OCLVersionMajor == '2')) {
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
          K.get_info<info::kernel_device_specific::work_group_size>(
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
        if (std::string(E.what()).find(
                "Total number of work-items in a work-group cannot exceed " +
                std::to_string(MaxKernelWGSize) + " for this kernel") ==
            std::string::npos) {
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
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeB failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Non-uniform work-groups are not allowed by default. "
                "Underlying OpenCL 2.x implementation supports this feature "
                "and to enable it, build device program with -cl-std=CL2.0") ==
            std::string::npos) {
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
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeC failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Non-uniform work-groups are not allowed by default. "
                "Underlying OpenCL 2.x implementation supports this feature "
                "and to enable it, build device program with -cl-std=CL2.0") ==
            std::string::npos) {
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

    // Local Size larger than Global.
    // CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified as larger
    // than the global size, then a different error string is used.
    // This is a sub-case of the more general 'non-uniform work group'
    {
      try {
        // parallel_for, 16 global, 17 local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeB2>(
              nd_range<1>(range<1>(16), range<1>(17)), [=](nd_item<1>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeB2 failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Local workgroup size greater than global range size. "
                "Non-uniform work-groups are not allowed by default. "
                "Underlying "
                "OpenCL 2.x implementation supports this feature and to enable "
                "it, build device program with -cl-std=CL2.0") ==
            std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeB2 failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeB2 failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr
            << "Test case OpenCL2XNegativeB2 failed: something unexpected "
               "has been caught"
            << std::endl;
        return 1;
      }

      // Local Size larger than Global, multi-dimensional
      // This is a sub-case of the more general 'non-uniform work group'
      try {
        // parallel_for, 6, 6, 6 global, 2, 2, 7 local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeC2>(
              nd_range<3>(range<3>(6, 6, 6), range<3>(2, 2, 7)),
              [=](nd_item<3>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeC2 failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Local workgroup size greater than global range size. "
                "Non-uniform work-groups are not allowed by default. "
                "Underlying OpenCL 2.x implementation supports this feature "
                "and to enable it, build device program with -cl-std=CL2.0") ==
            std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeC2 failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeC2 failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr
            << "Test case OpenCL2XNegativeC2 failed: something unexpected "
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
    // Multi-dimensional nd_range.
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
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeD failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Global work size {100, 1, 1} is not evenly divisible "
                "by local work-group size {3, 1, 1}. "
                "Non-uniform work-groups are not allowed by when "
                "-cl-uniform-work-group-size flag is used. Underlying "
                "OpenCL 2.x implementation supports this feature, but it is "
                "being "
                "disabled by -cl-uniform-work-group-size build flag") ==
            std::string::npos) {
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
    // Multi-dimensional nd_range.
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
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeE failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Global work size {16, 33, 100} is not evenly divisible by "
                "local work-group size {5, 3, 2}. "
                "Non-uniform work-groups are not allowed by when "
                "-cl-uniform-work-group-size flag is used. Underlying "
                "OpenCL 2.x implementation supports this feature, but it is "
                "being "
                "disabled by -cl-uniform-work-group-size build flag") ==
            std::string::npos) {
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

    // Local Size larger than Global, -cl-std=CL2.0 -cl-uniform-work-group-size
    // flag used CL_INVALID_WORK_GROUP_SIZE if local_work_size is specified as
    // larger than the global size, then a different error string is used. This
    // is a sub-case of the more general 'non-uniform work group'
    {
      program P(Q.get_context());
      P.build_with_kernel_type<class OpenCL2XNegativeD2>(
          "-cl-std=CL2.0 -cl-uniform-work-group-size");

      kernel K = P.get_kernel<class OpenCL2XNegativeD2>();
      try {
        // parallel_for, 16 global, 17 local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeD2>(
              K, nd_range<1>(range<1>(16), range<1>(17)), [=](nd_item<1>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeD2 failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Local work-group size {17, 1, 1} is greater than global range "
                "size {16, 1, 1}. "
                "Non-uniform work-groups are not allowed by when "
                "-cl-uniform-work-group-size flag is used. Underlying "
                "OpenCL 2.x implementation supports this feature, but it is "
                "being "
                "disabled by -cl-uniform-work-group-size build flag") ==
            std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeD2 failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeD2 failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr
            << "Test case OpenCL2XNegativeD2 failed: something unexpected "
               "has been caught"
            << std::endl;
        return 1;
      }
    }
    // Multi-dimensional nd_range.
    {
      program P(Q.get_context());
      P.build_with_kernel_type<class OpenCL2XNegativeE2>(
          "-cl-std=CL2.0 -cl-uniform-work-group-size");

      kernel K = P.get_kernel<class OpenCL2XNegativeE2>();
      try {
        // parallel_for, 6, 6, 6 global, 2, 2, 7 local -> fail
        Q.submit([&](handler &CGH) {
          CGH.parallel_for<class OpenCL2XNegativeE2>(
              K, nd_range<3>(range<3>(6, 6, 6), range<3>(2, 2, 7)),
              [=](nd_item<3>) {});
        });
        Q.wait_and_throw();
        // FIXME: some Intel runtimes contain bug and don't return expected
        // error code
        if (info::device_type::cpu != DeviceType ||
            DeviceVendorName.find("Intel") == std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeE2 failed: no exception has been "
                 "thrown\n";
          return 1; // We shouldn't be here, exception is expected
        }
      } catch (nd_range_error &E) {
        if (std::string(E.what()).find(
                "Local work-group size {7, 2, 2} is greater than global range "
                "size {6, 6, 6}. "
                "Non-uniform work-groups are not allowed by when "
                "-cl-uniform-work-group-size flag is used. Underlying "
                "OpenCL 2.x implementation supports this feature, but it is "
                "being "
                "disabled by -cl-uniform-work-group-size build flag") ==
            std::string::npos) {
          std::cerr
              << "Test case OpenCL2XNegativeE2 failed: unexpected exception: "
              << E.what() << std::endl;
          return 1;
        }
      } catch (runtime_error &E) {
        std::cerr
            << "Test case OpenCL2XNegativeE2 failed: unexpected exception: "
            << E.what() << std::endl;
        return 1;
      } catch (...) {
        std::cerr
            << "Test case OpenCL2XNegativeE2 failed: something unexpected "
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
