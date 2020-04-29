#include <CL/sycl.hpp>

#include <iostream>
#include <memory>
#include <vector>

constexpr const size_t DEFAULT_PROBLEM_SIZE = 16;

template <typename T>
struct WithOutputBuffer {
  WithOutputBuffer(size_t size) {
    _output_buffer_data.resize(size);
    _output_buffer.reset(new cl::sycl::buffer<T>(_output_buffer_data.data(), _output_buffer_data.size()));
  }

  WithOutputBuffer(const std::vector<T> &data) {
    _output_buffer_data = data;
    _output_buffer.reset(new cl::sycl::buffer<T>(_output_buffer_data.data(), _output_buffer_data.size()));
  }

  const std::vector<T> &getOutputBufferData() {
    // We cannoe access the data until the buffer is still alive
    _output_buffer.reset();
    return _output_buffer_data;
  }

  size_t getOutputBufferSize() const {
    return _output_buffer_data.size();
  }

protected:
  cl::sycl::buffer<T> &getOutputBuffer() {
    return *_output_buffer;
  }

  // Functor is being passed by-copy into cl::sycl::queue::submit and destroyed
  // one more time in there. We need to make sure that buffer is only released
  // once.
  std::shared_ptr<cl::sycl::buffer<T>> _output_buffer = nullptr;
  std::vector<T> _output_buffer_data;
};

template <typename T, size_t N>
struct WithInputBuffers {

  template <typename... Args>
  WithInputBuffers(Args... inputs) {
    static_assert(sizeof...(Args) == N, "All input buffers must be initialized");
    constructorHelper<0>(inputs...);
  }

  cl::sycl::buffer<T> &getInputBuffer(size_t i = 0) {
    return *_input_buffers[i];
  }

protected:
  std::shared_ptr<cl::sycl::buffer<T>> _input_buffers[N] = {nullptr};
  std::vector<T> _input_buffers_data[N];

private:
  template <int Index, typename... Args>
  void constructorHelper(const std::vector<T> &data, Args... rest) {
    _input_buffers_data[Index] = data;
    _input_buffers[Index].reset(new cl::sycl::buffer<T>(_input_buffers_data[Index].data(), _input_buffers_data[Index].size()));
    constructorHelper<Index + 1>(rest...);
  }

  template <int Index>
  void constructorHelper() {
    // nothing to do, recursion stop
  }
};

bool isInlineASMSupported(sycl::device Device) {

  sycl::string_class DriverVersion = Device.get_info<sycl::info::device::driver_version>();
  sycl::string_class DeviceVendorName = Device.get_info<sycl::info::device::vendor>();
  // TODO: query for some extension/capability/whatever once interface is
  // defined
  if (DeviceVendorName.find("Intel") == sycl::string_class::npos)
    return false;
  if (DriverVersion.length() < 5)
    return false;
  if (DriverVersion[2] != '.')
    return false;
  if (std::stoi(DriverVersion.substr(0, 2), nullptr, 10) < 20 || std::stoi(DriverVersion.substr(3, 2), nullptr, 10) < 12)
    return false;
  return true;
}

/// checks if device suppots inline asm feature and launches a test
///
/// \returns false if test wasn't launched (i.e.was skipped) and true otherwise
template <typename F>
bool launchInlineASMTest(F &f, bool requires_particular_sg_size = true) {
  try {
    cl::sycl::queue deviceQueue(cl::sycl::gpu_selector{});
    cl::sycl::device device = deviceQueue.get_device();

#if defined(INLINE_ASM)
    if (!isInlineASMSupported(device)) {
      std::cout << "Skipping test\n";
      return false;
    }
#endif

    if (requires_particular_sg_size && !device.has_extension("cl_intel_required_subgroup_size")) {
      std::cout << "Skipping test\n";
      return false;
    }

    deviceQueue.submit(f).wait();
  } catch (cl::sycl::exception &e) {
    std::cerr << "Caught exception: " << e.what() << std::endl;
  }
  return true;
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
