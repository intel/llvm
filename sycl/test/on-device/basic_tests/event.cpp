// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out

//==--------------- event.cpp - SYCL event test ----------------------------==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include <CL/sycl.hpp>
#include <iostream>

int main() {
  {
    std::cout << "move constructor" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event MovedEvent(std::move(Event));
    assert(hash == std::hash<cl::sycl::event>()(MovedEvent));
  }

  {
    std::cout << "move assignment operator" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event WillMovedEvent;
    WillMovedEvent = std::move(Event);
    assert(hash == std::hash<cl::sycl::event>()(WillMovedEvent));
  }

  {
    std::cout << "copy constructor" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event EventCopy(Event);
    assert(hash == std::hash<cl::sycl::event>()(Event));
    assert(hash == std::hash<cl::sycl::event>()(EventCopy));
    assert(Event == EventCopy);
  }

  {
    std::cout << "copy assignment operator" << std::endl;
    cl::sycl::event Event;
    size_t hash = std::hash<cl::sycl::event>()(Event);
    cl::sycl::event WillEventCopy;
    WillEventCopy = Event;
    assert(hash == std::hash<cl::sycl::event>()(Event));
    assert(hash == std::hash<cl::sycl::event>()(WillEventCopy));
    assert(Event == WillEventCopy);
  }

  // Check wait and wait_and_throw methods do not crash
  for (int i = 0; i < 4; ++i) {
    cl::sycl::queue Queue;
    float Data = 1.0;
    float Scalar = 2.0;
    cl::sycl::buffer<float, 1> Buf(&Data, cl::sycl::range<1>(1));
    auto Event = Queue.submit([&](cl::sycl::handler &cgh) {
      auto Acc = Buf.get_access<cl::sycl::access::mode::read_write>(cgh);

      cgh.single_task<class test_event>([=]() { Acc[0] = Scalar; });
    });

    switch (i) {
    case 0: {
      Event.wait();
      break;
    }
    case 1: {
      Event.wait_and_throw();
      break;
    }
    case 2: {
      cl::sycl::vector_class<cl::sycl::event> EventList = Event.get_wait_list();
      cl::sycl::event::wait(EventList);
      break;
    }
    case 3: {
      cl::sycl::vector_class<cl::sycl::event> EventList = Event.get_wait_list();
      cl::sycl::event::wait_and_throw(EventList);
      break;
    }
    }
  }
}
