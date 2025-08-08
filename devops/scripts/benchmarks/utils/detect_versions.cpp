#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>

#include <level_zero/ze_api.h>

#define _assert(cond, msg)                                                     \
  if (!(cond)) {                                                               \
    std::cout << std::endl << "Error: " << msg << std::endl;                   \
    exit(1);                                                                   \
  }

#define _success(res) res == ZE_RESULT_SUCCESS

std::string query_dpcpp_ver() { return std::string(__clang_version__); }

std::string query_l0_driver_ver() {
  // Initialize L0 drivers:
  ze_init_driver_type_desc_t driver_type = {};
  driver_type.stype = ZE_STRUCTURE_TYPE_INIT_DRIVER_TYPE_DESC;
  driver_type.flags = ZE_INIT_DRIVER_TYPE_FLAG_GPU;
  driver_type.pNext = nullptr;

  uint32_t driver_count = 0;
  ze_result_t result = zeInitDrivers(&driver_count, nullptr, &driver_type);
  _assert(_success(result), "Failed to initialize L0.");
  _assert(driver_count > 0, "No L0 drivers available.");

  std::vector<ze_driver_handle_t> drivers(driver_count);
  result = zeInitDrivers(&driver_count, drivers.data(), &driver_type);
  _assert(_success(result), "Could not fetch L0 drivers.");

  // Check support for fetching driver version strings:
  uint32_t ext_count = 0;
  result = zeDriverGetExtensionProperties(drivers[0], &ext_count, nullptr);
  _assert(_success(result), "Failed to obtain L0 extensions count.");
  _assert(ext_count > 0, "No L0 extensions available.");

  std::vector<ze_driver_extension_properties_t> extensions(ext_count);
  result =
      zeDriverGetExtensionProperties(drivers[0], &ext_count, extensions.data());
  _assert(_success(result), "Failed to obtain L0 extensions.");
  bool version_ext_support = false;
  for (const auto &extension : extensions) {
    // std::cout << extension.name << std::endl;
    if (strcmp(extension.name, "ZE_intel_get_driver_version_string")) {
      version_ext_support = true;
    }
  }
  _assert(version_ext_support,
          "ZE_intel_get_driver_version_string extension is not supported.");

  // Fetch L0 driver version:
  ze_result_t (*pfnGetDriverVersionFn)(ze_driver_handle_t, char *, size_t *);
  result = zeDriverGetExtensionFunctionAddress(drivers[0],
                                               "zeIntelGetDriverVersionString",
                                               (void **)&pfnGetDriverVersionFn);
  _assert(_success(result), "Failed to obtain GetDriverVersionString fn.");

  size_t ver_str_len = 0;
  result = pfnGetDriverVersionFn(drivers[0], nullptr, &ver_str_len);
  _assert(_success(result), "Call to GetDriverVersionString failed.");

  std::cout << "ver_str_len: " << ver_str_len << std::endl;
  ver_str_len++; // ver_str_len does not account for '\0'
  char *ver_str = (char *)calloc(ver_str_len, sizeof(char));
  result = pfnGetDriverVersionFn(drivers[0], ver_str, &ver_str_len);
  _assert(_success(result), "Failed to write driver version string.");

  std::string res(ver_str);
  free(ver_str);
  return res;
}

int main() {
  std::string dpcpp_ver = query_dpcpp_ver();
  std::cout << "DPCPP_VER='" << dpcpp_ver << "'" << std::endl;

  std::string l0_ver = query_l0_driver_ver();
  std::cout << "L0_VER='" << l0_ver << "'" << std::endl;
}
