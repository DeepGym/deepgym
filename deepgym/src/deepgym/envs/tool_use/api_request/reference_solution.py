"""Fetch the slideshow document at the URL given as argv[1] and print its title."""

import json
import sys
import urllib.request


def main():
    with urllib.request.urlopen(sys.argv[1], timeout=10) as response:
        data = json.load(response)
    print(data['slideshow']['title'])


if __name__ == '__main__':
    main()
