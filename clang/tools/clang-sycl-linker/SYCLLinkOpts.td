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

def llvm_spirv_options_EQ : Joined<["--", "-"], "llvm-spirv-options=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Options passed to llvm-spirv for LLVM IR to SPIR-V translation">;

def sycl_post_link_options_EQ : Joined<["--", "-"], "sycl-post-link-options=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Options passed to sycl-post-link tool">;

// Options to specify additional bitcode libraries linked with -only-needed.
def bitcode_library_EQ : Joined<["--"], "bitcode-library=">,
  Flags<[LinkerOnlyOption]>, MetaVarName<"<kind>-<triple>-<arch>=<path>">,
  HelpText<"Extra bitcode library to link">;

def no_canonical_prefixes : Flag<["--"], "no-canonical-prefixes">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Do not resolve symbolic links, turn relative paths into absolute ones, or do anything else to identify the executable">;

// Extra SYCL options to help generate sycl-post-link options that also depend
// on the target triple.
def sycl_remove_unused_external_funcs : Flag<["--", "-"], "sycl-remove-unused-external-funcs">,
  Flags<[LinkerOnlyOption, HelpHidden]>;
def no_sycl_remove_unused_external_funcs : Flag<["--", "-"], "no-sycl-remove-unused-external-funcs">,
  Flags<[LinkerOnlyOption, HelpHidden]>;
def sycl_device_code_split_esimd : Flag<["--", "-"], "sycl-device-code-split-esimd">,
  Flags<[LinkerOnlyOption, HelpHidden]>;
def no_sycl_device_code_split_esimd : Flag<["--", "-"], "no-sycl-device-code-split-esimd">,
  Flags<[LinkerOnlyOption, HelpHidden]>;
def sycl_add_default_spec_consts_image : Flag<["--", "-"], "sycl-add-default-spec-consts-image">,
  Flags<[LinkerOnlyOption, HelpHidden]>;
def no_sycl_add_default_spec_consts_image : Flag<["--", "-"], "no-sycl-add-default-spec-consts-image">,
  Flags<[LinkerOnlyOption, HelpHidden]>;

def sycl_thin_lto : Flag<["--", "-"], "sycl-thin-lto">,
Flags<[LinkerOnlyOption]>,  HelpText<"Link SYCL device code using thinLTO">;

// Options to enable/disable device dynamic linking.
def sycl_allow_device_image_dependencies : Flag<["--", "-"], "sycl-allow-device-image-dependencies">,
  Flags<[LinkerOnlyOption, HelpHidden]>,
  HelpText<"Allow dependencies between device code images">;

// Options to force the output to be of the SYCLBIN format.
def syclbin_EQ : Joined<["--", "-"], "syclbin=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"Output in the SYCLBIN binary format in the state specified by <arg> (input, object or executable)">;

// Options to specify SYCL device kernel bitcode input files, extracted from offload binaries.
def input_bitcode_file_EQ : Joined<["--", "-"], "input-bitcode-file=">,
  Flags<[LinkerOnlyOption]>,
  HelpText<"LLVM bitcode file containing compiled SYCL device kernels to be linked">;