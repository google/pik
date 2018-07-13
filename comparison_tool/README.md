This is a Web UI for comparing image codecs side by side.

It is based on revision [`4a057`](https://github.com/WyohKnott/image-formats-comparison/tree/4a05707613d21dad117394e706490895f96e3b45)
of https://github.com/WyohKnott/image-formats-comparison.

It requires a `comparisonfiles.json` file and a `comparisonfiles` folder
containing the sets of images to compare. They can be generated using
https://github.com/WyohKnott/image-comparison-sources (or taken from the
original viewer).

The main difference with the original is that in addition to showing the output
of two different codecs on each side, it also shows the original image in the
middle.

Since the viewer gets its files via HTTP requests, it should be served by a Web
server. A convenient option for quickly running it locally is to execute the
following command from this directory:

```bash
python3 -m http.server
```
