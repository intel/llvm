// RUN: %{build} -o %t.out
// RUN: %{run} %t.out

#include <sycl/detail/core.hpp>

int main() {
  int Failures = 0;

  // Case 1 - Event wait_and_throw with custom handler on queue.
  {
    bool Caught = false;
    auto AHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 1 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::queue Q{sycl::default_selector_v, AHandler};
    Q.submit([&](sycl::handler &CGH) {
       CGH.host_task([=]() { throw std::runtime_error("Case 1"); });
     }).wait_and_throw();
    if (!Caught) {
      std::cout << "Case 1 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 2 - Event wait_and_throw with custom handler on context.
  {
    bool Caught = false;
    auto AHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 2 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::context Ctx{AHandler};
    sycl::queue Q{Ctx, sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
       CGH.host_task([=]() { throw std::runtime_error("Case 2"); });
     }).wait_and_throw();
    if (!Caught) {
      std::cout << "Case 2 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 3 - Event wait_and_throw with custom handler on both queue and
  //          context.
  {
    bool Caught = false;
    auto CtxAHandler = [&](sycl::exception_list EL) {
      std::cout << "Case 3 - Unexpected handler used." << std::endl;
      ++Failures;
    };
    auto QAHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 3 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::context Ctx{CtxAHandler};
    sycl::queue Q{Ctx, sycl::default_selector_v, QAHandler};
    Q.submit([&](sycl::handler &CGH) {
       CGH.host_task([=]() { throw std::runtime_error("Case 3"); });
     }).wait_and_throw();
    if (!Caught) {
      std::cout << "Case 3 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 4 - Queue wait_and_throw with custom handler on queue.
  {
    bool Caught = false;
    auto AHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 1 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::queue Q{sycl::default_selector_v, AHandler};
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() { throw std::runtime_error("Case 4"); });
    });
    Q.wait_and_throw();
    if (!Caught) {
      std::cout << "Case 4 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 5 - Queue wait and throw_asynchronous with custom handler on queue.
  {
    bool Caught = false;
    auto AHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 5 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::queue Q{sycl::default_selector_v, AHandler};
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() { throw std::runtime_error("Case 5"); });
    });
    Q.wait();
    Q.throw_asynchronous();
    if (!Caught) {
      std::cout << "Case 5 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 6 - Queue wait_and_throw with custom handler on context.
  {
    bool Caught = false;
    auto AHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 6 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::context Ctx{AHandler};
    sycl::queue Q{Ctx, sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() { throw std::runtime_error("Case 6"); });
    });
    Q.wait_and_throw();
    if (!Caught) {
      std::cout << "Case 6 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 7 - Queue wait and throw_asynchronous with custom handler on context.
  {
    bool Caught = false;
    auto AHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 7 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::context Ctx{AHandler};
    sycl::queue Q{Ctx, sycl::default_selector_v};
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() { throw std::runtime_error("Case 7"); });
    });
    Q.wait();
    Q.throw_asynchronous();
    if (!Caught) {
      std::cout << "Case 7 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 8 - Queue wait_and_throw with custom handler on both queue and
  //          context.
  {
    bool Caught = false;
    auto CtxAHandler = [&](sycl::exception_list EL) {
      std::cout << "Case 8 - Unexpected handler used." << std::endl;
      ++Failures;
    };
    auto QAHandler = [&](sycl::exception_list EL) {
      Caught = true;
      if (EL.size() != 1) {
        std::cout << "Case 8 - Unexpected number of exceptions." << std::endl;
        ++Failures;
      }
    };
    sycl::context Ctx{CtxAHandler};
    sycl::queue Q{Ctx, sycl::default_selector_v, QAHandler};
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() { throw std::runtime_error("Case 8"); });
    });
    Q.wait_and_throw();
    if (!Caught) {
      std::cout << "Case 8 - Async exception not caught by handler."
                << std::endl;
      ++Failures;
    }
  }

  // Case 9 - Queue dying without having consumed its asynchronous exceptions.
  {
    auto CtxAHandler = [&](sycl::exception_list EL) {
      std::cout << "Case 9 - Unexpected context handler used." << std::endl;
      ++Failures;
    };
    auto QAHandler = [&](sycl::exception_list EL) {
      std::cout << "Case 9 - Unexpected queue handler used." << std::endl;
      ++Failures;
    };
    sycl::context Ctx{CtxAHandler};
    sycl::queue Q{Ctx, sycl::default_selector_v, QAHandler};
    Q.submit([&](sycl::handler &CGH) {
      CGH.host_task([=]() { throw std::runtime_error("Case 9"); });
    });
  }

  return Failures;
}
