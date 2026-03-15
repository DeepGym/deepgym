"""Fetch JSON from httpbin.org and extract the slideshow title."""

import json
import urllib.request


def main():
    url = 'https://httpbin.org/json'
    response = urllib.request.urlopen(url)
    data = json.loads(response.read().decode('utf-8'))
    title = data['slideshow']['title']
    print(title)


if __name__ == '__main__':
    main()
