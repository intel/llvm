#include <cstring>
#include <uur/environment.h>
#include <uur/utils.h>

namespace uur {

PlatformEnvironment *PlatformEnvironment::instance = nullptr;

std::ostream &operator<<(std::ostream &out,
                         const ur_platform_handle_t &platform) {
  size_t size;
  urPlatformGetInfo(platform, UR_PLATFORM_INFO_NAME, 0, nullptr, &size);
  std::vector<char> name(size);
  urPlatformGetInfo(platform, UR_PLATFORM_INFO_NAME, size, name.data(),
                    nullptr);
  out << name.data();
  return out;
}

std::ostream &operator<<(std::ostream &out,
                         const std::vector<ur_platform_handle_t> &platforms) {
  for (auto platform : platforms) {
    out << "\n  * \"" << platform << "\"";
  }
  return out;
}

uur::PlatformEnvironment::PlatformEnvironment(int argc, char **argv)
    : platform_options{parsePlatformOptions(argc, argv)} {
  instance = this;
  ur_platform_init_flags_t platform_flags = 0;
  ur_device_init_flags_t device_flags = 0;
  if (urInit(platform_flags, device_flags)) {
    error = "urInit() failed";
    return;
  }

  uint32_t count = 0;
  if (urPlatformGet(0, nullptr, &count)) {
    error = "urPlatformGet() failed to get number of platforms.";
    return;
  }

  if (count == 0) {
    error = "Failed to find any platforms.";
    return;
  }

  std::vector<ur_platform_handle_t> platforms(count);
  if (urPlatformGet(count, platforms.data(), nullptr)) {
    error = "urPlatformGet failed to get platforms.";
    return;
  }

  if (platform_options.platform_name.empty()) {
    if (platforms.size() == 1) {
      platform = platforms[0];
    } else {
      std::stringstream ss_error;
      ss_error
          << "Select a single platform from below using the --platform=NAME "
             "command-line option:"
          << platforms;
      error = ss_error.str();
      return;
    }
  } else {
    for (auto candidate : platforms) {
      size_t size;
      if (urPlatformGetInfo(candidate, UR_PLATFORM_INFO_NAME, 0, nullptr,
                            &size)) {
        error = "urPlatformGetInfoFailed";
        return;
      }
      std::vector<char> platform_name(size);
      if (urPlatformGetInfo(candidate, UR_PLATFORM_INFO_NAME, size,
                            platform_name.data(), nullptr)) {
        error = "urPlatformGetInfo() failed";
        return;
      }
      if (platform_options.platform_name == platform_name.data()) {
        platform = candidate;
        break;
      }
    }
    if (!platform) {
      std::stringstream ss_error;
      ss_error << "Platform \"" << platform_options.platform_name
               << "\" not found. Select a single platform from below using the "
                  "--platform=NAME command-line options:"
               << platforms;
      error = ss_error.str();
      return;
    }
  }
};

void uur::PlatformEnvironment::SetUp() {
  if (!error.empty()) {
    FAIL() << error;
  }
}

void uur::PlatformEnvironment::TearDown() {
  ur_tear_down_params_t tear_down_params{};
  if (urTearDown(&tear_down_params)) {
    FAIL() << "urTearDown() failed";
  }
}

PlatformEnvironment::PlatformOptions
PlatformEnvironment::parsePlatformOptions(int argc, char **argv) {
  PlatformOptions options;
  for (int argi = 1; argi < argc; ++argi) {
    const char *arg = argv[argi];
    if (!(std::strcmp(arg, "-h") && std::strcmp(arg, "--help"))) {
      // TODO - print help
      break;
    } else if (std::strncmp(arg, "--platform=", sizeof("--platform=") - 1) ==
               0) {
      options.platform_name = std::string(&arg[std::strlen("--platform=")]);
    }
  }
  return options;
}

DevicesEnvironment *DevicesEnvironment::instance = nullptr;

DevicesEnvironment::DevicesEnvironment(int argc, char **argv)
    : PlatformEnvironment(argc, argv) {
  instance = this;
  uint32_t count = 0;
  if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, 0, nullptr, &count)) {
    error = "urDevicesGet() failed to get number of devices.";
    return;
  }
  if (count == 0) {
    error = "Could not find any devices associated with the platform";
    return;
  }
  devices.resize(count);
  if (urDeviceGet(platform, UR_DEVICE_TYPE_ALL, count, devices.data(),
                  nullptr)) {
    error = "urDeviceGet() failed to get devices.";
    return;
  }
};

void DevicesEnvironment::SetUp() {
  PlatformEnvironment::SetUp();
  if (devices.empty() || !error.empty()) {
    FAIL() << error;
  }
}
} // namespace uur
