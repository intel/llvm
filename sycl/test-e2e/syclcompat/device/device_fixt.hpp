/***************************************************************************
 *
 *  Copyright (C) Codeplay Software Ltd.
 *
 *  Part of the LLVM Project, under the Apache License v2.0 with LLVM
 *  Exceptions. See https://llvm.org/LICENSE.txt for license information.
 *  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCLcompat
 *
 *  device_fixt.h
 *
 *  Description:
 *     Fixture helpers for to tests the extended device functionality
 **************************************************************************/

#pragma once

#include <sycl/detail/core.hpp>
#include <syclcompat/device.hpp>

class DeviceTestsFixt {
protected:
  unsigned int n_devices{};
  sycl::queue def_q_;

public:
  DeviceTestsFixt()
      : n_devices{syclcompat::device_count()},
        def_q_{syclcompat::get_default_queue()} {}

  unsigned int get_n_devices() { return n_devices; }
  sycl::queue get_queue() { return def_q_; }
};

class DeviceExtFixt {
protected:
  syclcompat::device_ext &dev_;

public:
  DeviceExtFixt() : dev_{syclcompat::get_current_device()} { SetUp(); }

  void SetUp() { dev_.reset(); }

  syclcompat::device_ext &get_dev_ext() { return dev_; }
};

// Helper for counting the output lines of syclcompat::list_devices
// Used to override std::cout
class CountingStream : public std::streambuf {
public:
  CountingStream(std::streambuf *buf) : buf(buf), line_count(0) {}

  int overflow(int c) override {
    if (c == '\n') {
      ++line_count;
    }
    return buf->sputc(c);
  }

  std::streamsize xsputn(const char_type *s, std::streamsize count) override {
    for (std::streamsize i = 0; i < count; ++i) {
      if (s[i] == '\n') {
        ++line_count;
      }
    }
    return buf->sputn(s, count);
  }

  int get_line_count() const { return line_count; }

private:
  std::streambuf *buf;
  int line_count;
};
