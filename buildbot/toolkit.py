import subprocess
import time


def cmd_run(cmd, env=None, cwd=None, shell=False, out=None, err=None):

    print(f"[command]: {subprocess.list2cmdline(cmd)}")
    try:
        completed_process = subprocess.run(cmd, shell=shell, env=env, cwd=cwd, check=True,
                                           stdout=out, stderr=err, encoding="utf-8", errors="backslashreplace")
        return completed_process.returncode, completed_process.stdout
    except subprocess.CalledProcessError as failed_process:
        return failed_process.returncode, failed_process.stdout


def cmd_run_with_retry(cmd, env=None, cwd=None, shell=False, out=None, err=None, catch_out=False, maxtry=5, interval=30):

    if maxtry < 1:
        raise Exception("Parameter \"maxtry\" must be a positive integer.")

    if catch_out == True:
        out = subprocess.PIPE
        err = subprocess.STDOUT

    count = maxtry
    while count:
        returncode, stdout = cmd_run(cmd, shell=shell, env=env, cwd=cwd, out=out, err=err)
        if returncode == 0:
            break
        print("Failed to execute command")
        time.sleep(30)
        count -= 1

    return returncode, stdout
