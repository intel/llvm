import os
import re
import sys
import json
import urllib
import tempfile
import subprocess
from urllib import request
from pathlib import Path
import argparse

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from options import options


def _get_patch_from_ver(ver: str) -> str:
    """Extract patch from a version string."""
    # L0 version strings follows semver: major.minor.patch+optional
    # compute-runtime version tags follow year.WW.patch.optional instead,
    # but both follow a quasi-semver versioning where the patch, optional
    # is still the same across both version string formats.
    patch = re.sub(r"^\d+\.\d+\.", "", ver)
    patch = re.sub(r"\+", ".", patch, count=1)
    return patch


class DetectVersion:
    _instance = None

    def __init__(self):
        raise RuntimeError("Use init() to init and instance() to get instead.")

    @classmethod
    def init(cls, detect_ver_path: Path, dpcpp_exec: str = "clang++"):
        """
        Constructs the singleton instance for DetectVersion, and initializes by
        building and run detect_version.cpp, which outputs:
          - L0 driver version via ZE_intel_get_driver_version_string extension,
          - DPC++ version via `__clang_version__` builtin.

        Remind: DO NOT allow user input in args.

        Parameters:
        detect_ver_path (Path): Path to detect_version.cpp
        dpcpp_exec (str): Name of DPC++ executable
        """
        if cls._instance is not None:
            return cls._instance

        detect_ver_exe = tempfile.mktemp()
        result = subprocess.run(
            [dpcpp_exec, "-lze_loader", detect_ver_path, "-o", detect_ver_exe],
            check=True,
            env=os.environ,
        )
        result = subprocess.run(
            [detect_ver_exe],
            check=True,
            text=True,
            capture_output=True,
            env=os.environ,
        )
        # Variables are printed to stdout, each var is on its own line
        result_vars = result.stdout.strip().split("\n")

        def get_var(var_name: str) -> str:
            var_str = next(
                filter(lambda v: re.match(f"^{var_name}='.*'", v), result_vars)
            )
            return var_str[len(f"{var_name}='") : -len("'")]

        cls._instance = cls.__new__(cls)
        cls._instance.l0_ver = get_var("L0_VER")
        cls._instance.dpcpp_ver = get_var("DPCPP_VER")
        cls._instance.dpcpp_exec = dpcpp_exec

        # Populate the computer-runtime version string cache: Since API calls
        # are expensive, we want to avoid API calls when possible, i.e.:
        # - Avoid a second API call if compute_runtime_ver was already obtained
        # - Avoid an API call altogether if the user provides a valid
        #   COMPUTE_RUNTIME_TAG_CACHE environment variable.
        cls._instance.compute_runtime_ver_cache = None
        l0_ver_patch = _get_patch_from_ver(get_var("L0_VER"))
        env_cache_ver = os.getenv("COMPUTE_RUNTIME_TAG_CACHE", default="")
        env_cache_patch = _get_patch_from_ver(env_cache_ver)
        # L0 patch often gets padded with 0's: if the environment variable
        # matches up with the prefix of the l0 version patch, the cache is
        # indeed referring to the same version.
        if env_cache_patch == l0_ver_patch[: len(env_cache_patch)]:
            print(
                f"Using compute_runtime tag from COMPUTE_RUNTIME_TAG_CACHE: {env_cache_var}"
            )
            cls._instance.compute_runtime_ver_cache = env_cache_ver

        return cls._instance

    @classmethod
    def instance(cls):
        """
        Returns singleton instance of DetectVersion if it has been initialized
        via init(), otherwise return None.
        """
        return cls._instance

    def get_l0_ver(self) -> str:
        """
        Returns the full L0 version string.
        """
        return self.l0_ver

    def get_dpcpp_ver(self) -> str:
        """
        Returns the full DPC++ version / clang version string of DPC++ used.
        """
        return self.dpcpp_ver

    def get_dpcpp_git_info(self) -> [str, str]:
        """
        Returns: (git_repo, commit_hash)
        """
        # clang++ formats are in <clang ver> (<git url> <commit>): if this
        # regex does not match, it is likely this is not upstream clang.
        git_info_match = re.search(r"\(http.+ [0-9a-f]+\)", self.dpcpp_ver)
        if git_info_match is None:
            raise RuntimeError(
                f"detect_version: Unable to obtain git info from {self.dpcpp_exec}, are you sure you are using DPC++?"
            )
        git_info = git_info_match.group(0)
        return git_info[1:-1].split(" ")

    def get_dpcpp_commit(self) -> str:
        git_info = self.get_dpcpp_git_info()
        if git_info is None:
            return options.detect_versions.not_found_placeholder
        return git_info[1]

    def get_dpcpp_repo(self) -> str:
        git_info = self.get_dpcpp_git_info()
        if git_info is None:
            return options.detect_versions.not_found_placeholder
        return git_info[0]

    def get_compute_runtime_ver_cached(self) -> str:
        return self.compute_runtime_ver_cache

    def get_compute_runtime_ver(self) -> str:
        """
        Returns the compute-runtime version by deriving from l0 version.
        """
        if self.compute_runtime_ver_cache is not None:
            return self.compute_runtime_ver_cache

        patch = _get_patch_from_ver(self.l0_ver)

        # TODO unauthenticated users only get 60 API calls per hour: this will
        # not work if we enable benchmark CI in precommit.
        url = options.detect_versions.compute_runtime_tag_api

        print(f"Fetching compute-runtime tag from {url}...")
        try:
            for _ in range(options.detect_versions.max_api_calls):
                res = request.urlopen(url)
                tags = [tag["name"] for tag in json.loads(res.read())]

                for tag in tags:
                    tag_patch = _get_patch_from_ver(tag)
                    # compute-runtime's cmake files produces "optional" fields
                    # padded with 0's: this means e.g. L0 version string
                    # 1.6.32961.200000 could be either compute-runtime ver.
                    # 25.09.32961.2, 25.09.32961.20, or even 25.09.32961.200.
                    #
                    # Thus, we take the longest match. Since the github api
                    # provides tags from newer -> older, we take the first tag
                    # that matches as it would be the "longest" ver. to match.
                    if tag_patch == patch[: len(tag_patch)]:
                        self.compute_runtime_ver_cache = tag
                        return tag

                def get_link_name(link: str) -> str:
                    rel_str = re.search(r'rel="\w+"', link).group(0)
                    return rel_str[len('rel="') : -len('"')]

                def get_link_url(link: str) -> str:
                    return link[link.index("<") + 1 : link.index(">")]

                links = {
                    get_link_name(link): get_link_url(link)
                    for link in res.getheader("Link").split(", ")
                }

                if "next" in links:
                    url = links["next"]
                else:
                    break

        except urllib.error.HTTPError as e:
            print(f"HTTP error {e.code}: {e.read().decode('utf-8')}")

        except urllib.error.URLError as e:
            print(f"URL error: {e.reason}")

        print(f"WARNING: unable to find compute-runtime version")
        return options.detect_versions.not_found_placeholder


def main(components: [str]):
    detect_res = DetectVersion.init(f"{os.path.dirname(__file__)}/detect_versions.cpp")

    str2fn = {
        "dpcpp_repo": detect_res.get_dpcpp_repo,
        "dpcpp_commit": detect_res.get_dpcpp_commit,
        "l0_ver": detect_res.get_l0_ver,
        "compute_runtime_ver": detect_res.get_compute_runtime_ver,
    }

    def remove_undefined_components(component: str) -> bool:
        if component not in str2fn:
            print(f"# Warn: unknown component: {component}", file=sys.stderr)
            return False
        return True

    components_clean = filter(remove_undefined_components, components)

    for s in map(lambda c: f"{c.upper()}={str2fn[c]()}", components_clean):
        print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Get version information for specified components."
    )
    parser.add_argument(
        "components",
        type=str,
        help="""
		Comma-separated list of components to get version information for.
		Valid options: dpcpp_repo,dpcpp_commit,l0_ver,compute_runtime_ver
		""",
    )
    args = parser.parse_args()

    main(map(lambda c: c.strip(), args.components.split(",")))
