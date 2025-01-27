#include <sycl/detail/core.hpp>

#include <iostream>
#include <memory>
#include <vector>

constexpr const size_t DEFAULT_PROBLEM_SIZE = 16;

template <typename T> struct WithOutputBuffer {
  WithOutputBuffer(size_t size) {
    _output_buffer_data.resize(size);
    _output_buffer.reset(new sycl::buffer<T>(_output_buffer_data.data(),
                                             _output_buffer_data.size()));
  }

  WithOutputBuffer(const std::vector<T> &data) {
    _output_buffer_data = data;
    _output_buffer.reset(new sycl::buffer<T>(_output_buffer_data.data(),
                                             _output_buffer_data.size()));
  }

  const std::vector<T> &getOutputBufferData() {
    // We cannoe access the data until the buffer is still alive
    _output_buffer.reset();
    return _output_buffer_data;
  }

  size_t getOutputBufferSize() const { return _output_buffer_data.size(); }

protected:
  sycl::buffer<T> &getOutputBuffer() { return *_output_buffer; }

  // Functor is being passed by-copy into sycl::queue::submit and destroyed
  // one more time in there. We need to make sure that buffer is only released
  // once.
  std::shared_ptr<sycl::buffer<T>> _output_buffer = nullptr;
  std::vector<T> _output_buffer_data;
};

template <typename T, size_t N> struct WithInputBuffers {

  template <typename... Args> WithInputBuffers(Args... inputs) {
    static_assert(sizeof...(Args) == N,
                  "All input buffers must be initialized");
    constructorHelper<0>(inputs...);
  }

  sycl::buffer<T> &getInputBuffer(size_t i = 0) { return *_input_buffers[i]; }

protected:
  std::shared_ptr<sycl::buffer<T>> _input_buffers[N] = {nullptr};
  std::vector<T> _input_buffers_data[N];

private:
  template <int Index, typename... Args>
  void constructorHelper(const std::vector<T> &data, Args... rest) {
    _input_buffers_data[Index] = data;
    _input_buffers[Index].reset(new sycl::buffer<T>(
        _input_buffers_data[Index].data(), _input_buffers_data[Index].size()));
    _input_buffers[Index]->set_final_data(nullptr);
    constructorHelper<Index + 1>(rest...);
  }

  template <int Index> void constructorHelper() {
    // nothing to do, recursion stop
  }
};

bool isInlineASMSupported(sycl::device Device) {

  std::string DriverVersion =
      Device.get_info<sycl::info::device::driver_version>();
  std::string DeviceVendorName = Device.get_info<sycl::info::device::vendor>();
  // TODO: query for some extension/capability/whatever once interface is
  // defined
  if (DeviceVendorName.find("Intel") == std::string::npos)
    return false;

  return true;
}

auto exception_handler = [](sycl::exception_list exceptions) {
  for (std::exception_ptr const &e : exceptions) {
    try {
      std::rethrow_exception(e);
    } catch (sycl::exception const &e) {
      std::cout << "Caught asynchronous SYCL exception:\n"
                << e.what() << std::endl;
    }
  }
};

template <typename F>
bool launchInlineASMTestImpl(F &f,
                             const std::vector<int> &RequiredSGSizes = {}) {
  sycl::queue deviceQueue(sycl::gpu_selector_v, exception_handler);
  sycl::device device = deviceQueue.get_device();

  if (!isInlineASMSupported(device)) {
    std::cout << "Skipping test\n";
    return false;
  }

  if (!RequiredSGSizes.empty()) {
    auto supported_sg_sizes =
        device.get_info<sycl::info::device::sub_group_sizes>();
    if (std::any_of(RequiredSGSizes.begin(), RequiredSGSizes.end(),
                    [&](size_t RequiredSGSize) {
                      return std::find(supported_sg_sizes.begin(),
                                       supported_sg_sizes.end(),
                                       RequiredSGSize) ==
                             supported_sg_sizes.end();
                    })) {
      std::cout << "Skipping test\n";
      return false;
    }
  }

  deviceQueue.submit(f).wait_and_throw();

  return true;
}

/// checks if device suppots inline asm feature and launches a test
///
/// \returns false if test wasn't launched (i.e.was skipped) and true otherwise
template <typename F>
bool launchInlineASMTest(F &f, const std::vector<int> &RequiredSGSizes = {},
                         bool exception_expected = false) {
  bool result = false;
  try {
    result = launchInlineASMTestImpl(f, RequiredSGSizes);
  } catch (sycl::exception &e) {
    if (exception_expected && e.code() == sycl::errc::build) {
      std::cout << "Caught expected exception." << std::endl;
    } else {
      std::cout << "Caught unexpected exception." << std::endl;
      throw e;
    }
  }
  return result;
}

template <typename T>
bool verify_all_the_same(const std::vector<T> &input, T reference_value) {
  for (int i = 0; i < input.size(); ++i)
    if (input[i] != reference_value) {
      std::cerr << "At index: " << i << " ";
      std::cerr << input[i] << " != " << reference_value << "\n";
      return false;
    }
  return true;
}
