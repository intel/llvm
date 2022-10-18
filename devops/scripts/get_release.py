from urllib.request import urlopen
import json
import sys

def get_release_by_tag(repo, tag):
    release = urlopen("https://api.github.com/repos/" + repo + "/releases/tags/" + tag).read()
    return json.loads(release)

def get_latest_release(repo):
    release = urlopen("https://api.github.com/repos/" + repo + "/releases/latest").read()
    return json.loads(release)

repo = sys.argv[1]
tag = sys.argv[2]

if tag == "latest":
    release = get_latest_release(repo)
else:
    release = get_release_by_tag(repo, tag)

for item in release["assets"]:
    print(item["browser_download_url"])

