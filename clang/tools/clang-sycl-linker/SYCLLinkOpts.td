include "llvm/Option/OptParser.td"

def LinkerOnlyOption : OptionFlag;

def help : Flag<["-", "--"], "help">,
  HelpText<"Display available options (--help-hidden for more)">;

def help_hidden : Flag<["-", "--"], "help-hidden">,
  HelpText<"Display all available options">;

def verbose : Flag<["-"], "v">, HelpText<"Print verbose information">;
def version : Flag<["--"], "version">,
  HelpText<"Display the version number and exit">;

def o : JoinedOrSeparate<["-"], "o">, MetaVarName<"<path>">,
  HelpText<"Path to file to write output">;
def output : Separate<["--"], "output-file">, Alias<o>, Flags<[HelpHidden]>,
  HelpText<"Alias for -o">;

def library_path_EQ : Joined<["--", "-"], "library-path=">,
  Flags<[HelpHidden]>, HelpText<"Add <dir> to the library search path">;

def device_libs_EQ : CommaJoined<["--", "-"], "device-libs=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"A comma separated list of device libraries that are linked during the device link.">;

def arch_EQ : Joined<["--", "-"], "arch=">,
              Flags<[LinkerOnlyOption]>,
              MetaVarName<"<arch>">,
              HelpText<"The device architecture">;
def triple_EQ : Joined<["--", "-"], "triple=">,
                Flags<[LinkerOnlyOption]>,
                MetaVarName<"<triple>">,
                HelpText<"The device target triple">;

def save_temps : Flag<["--", "-"], "save-temps">,
  Flags<[LinkerOnlyOption]>, HelpText<"Save intermediate results">;

def dry_run : Flag<["--", "-"], "dry-run">, Flags<[LinkerOnlyOption]>,
  HelpText<"Print generated commands without running.">;

def spirv_dump_device_code_EQ : Joined<["--", "-"], "spirv-dump-device-code=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Path to the folder where the tool dumps SPIR-V device code. Other formats aren't dumped.">;

def print_linked_module : Flag<["--"], "print-linked-module">,
                          Flags<[LinkerOnlyOption]>,
                          HelpText<"Print the linked module's IR for testing">;

def ocloc_options_EQ : Joined<["--", "-"], "ocloc-options=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Options passed to ocloc for Intel GPU AOT compilation">;

def opencl_aot_options_EQ : Joined<["--", "-"], "opencl-aot-options=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Options passed to opencl-aot for Intel CPU AOT compilation">;

// SYCL post-link options
def sycl_post_link_options_EQ : Joined<["--", "-"], "sycl-post-link-options=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Options passed to sycl-post-link tool">;

def sycl_module_split_mode_EQ : Joined<["--", "-"], "sycl-module-split-mode=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Module split mode: kernel, source, auto, or none">;

def use_sycl_post_link_tool : Flag<["--", "-"], "use-sycl-post-link-tool">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Use external sycl-post-link tool (default)">;

def no_use_sycl_post_link_tool : Flag<["--", "-"], "no-use-sycl-post-link-tool">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Use sycl-post-link library API instead of tool">;

def sycl_device_code_split_esimd : Flag<["--", "-"], "sycl-device-code-split-esimd">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Enable ESIMD code splitting">;

def no_sycl_device_code_split_esimd : Flag<["--", "-"], "no-sycl-device-code-split-esimd">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Disable ESIMD code splitting">;

def sycl_add_default_spec_consts_image : Flag<["--", "-"], "sycl-add-default-spec-consts-image">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Generate module descriptor with default specialization constants">;

def no_sycl_add_default_spec_consts_image : Flag<["--", "-"], "no-sycl-add-default-spec-consts-image">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Do not generate default spec consts module descriptor">;

def sycl_remove_unused_external_funcs : Flag<["--", "-"], "sycl-remove-unused-external-funcs">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Remove unused external functions">;

def no_sycl_remove_unused_external_funcs : Flag<["--", "-"], "no-sycl-remove-unused-external-funcs">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Keep all external functions (default)">;

def sycl_allow_device_image_dependencies : Flag<["--", "-"], "sycl-allow-device-image-dependencies">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Allow dependencies between device images">;

def sycl_thin_lto : Flag<["--", "-"], "sycl-thin-lto">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Enable thin LTO for SYCL device code">;

def syclbin_EQ : Joined<["--", "-"], "syclbin=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Generate SYCLBIN output (input, object, or executable)">;

def sycl_device_library_location_EQ : CommaJoined<["--", "-"], "sycl-device-library-location=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Path(s) to SYCL device library directory">;
