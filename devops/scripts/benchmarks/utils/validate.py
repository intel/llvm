import re


def validate_on_re(val: str, regex: re.Pattern, throw: Exception = None):
    """
    Returns True if val is matched by pattern defined by regex, otherwise False.

    If `throw` argument is not None: return val as-is if val matches regex,
    otherwise raise error defined by throw.
    """
    is_matching: bool = re.compile(regex).match(val.strip()) is not None

    if throw is None:
        return is_matching
    elif not is_matching:
        raise throw
    else:
        return val


class Validate:
    """Static class containing methods for validating various fields"""

    @staticmethod
    def runner_name(runner_name: str, throw: Exception = None):
        """
        Returns True if runner_name is clean (no illegal characters).
        """
        return validate_on_re(runner_name, r"^[a-zA-Z0-9_]+$", throw=throw)

    @staticmethod
    def save_name(save: str, throw: Exception = None):
        """
        Returns True if save is within [a-zA-Z0-9_-].

        If throw argument is specified: return save as is if save satisfies
        aforementioned regex, otherwise raise error defined by throw.
        """
        return validate_on_re(save, r"^[a-zA-Z0-9_-]+$", throw=throw)

    @staticmethod
    def timestamp(t: str, throw: Exception = None):
        """
        Returns True if t is in form YYYYMMDD_HHMMSS, otherwise False.

        If throw argument is specified: return t as-is if t is in aforementioned
        format, otherwise raise error defined by throw.
        """
        return validate_on_re(
            t,
            r"^\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[01])_([01][0-9]|2[0-3])[0-5][0-9][0-5][0-9]$",
            throw=throw,
        )

    @staticmethod
    def github_repo(repo: str, throw: Exception = None):
        """
        Returns True if repo is of form <owner>/<repo name>

        If throw argument is specified: return repo as-is if repo is in
        aforementioned format, otherwise raise error defined by throw.
        """
        return validate_on_re(
            re.sub(r"^https?://github.com/", "", repo),
            r"^[a-zA-Z0-9_-]{1,39}/[a-zA-Z0-9_.-]{1,100}$",
            throw=throw,
        )

    @staticmethod
    def commit_hash(commit: str, throw: Exception = None, trunc: int = 40):
        """
        Returns True if commit is a valid git commit hash.

        If throw argument is specified: return commit hash (truncated to trunc
        chars long) if commit is a valid commit hash, otherwise raise error
        defined by throw.
        """
        commit_re = r"^[a-f0-9]{7,40}$"
        if throw is None:
            return validate_on_re(commit, commit_re)
        else:
            return validate_on_re(commit, commit_re, throw=throw)[:trunc]
