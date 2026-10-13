# API Request

Make an HTTP GET request and extract data from the JSON response.

## Instructions

Write a Python script that:

1. Takes the API URL as its first command-line argument (`sys.argv[1]`).
2. Makes a GET request to that URL.
3. Parses the JSON response, a document shaped like
   `{"slideshow": {"title": "...", "author": "...", "slides": [{"title": "...", ...}]}, ...}`.
4. Prints the `slideshow.title` field to stdout, and nothing else.

The script is run against several documents whose fields, values and key order
vary. Use only the Python standard library.
