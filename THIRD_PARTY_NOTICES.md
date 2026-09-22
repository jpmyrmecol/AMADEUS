# Third-Party Notices

## CustomTkinter theme

`assets/deep_green.json` is based on the `green.json` theme included with
CustomTkinter and has been modified for AMADEUS.

CustomTkinter is licensed under the MIT License.

Copyright (c) 2023 Tom Schimansky

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## FFmpeg

AMADEUS inspects and, when the user agrees, converts input videos with FFmpeg.
The FFmpeg build is **pinned**, not discovered on the host: it is the binary
shipped inside the `imageio-ffmpeg` wheel that `pyproject.toml` fixes at
`imageio-ffmpeg==0.6.0` and that `uv.lock` pins by SHA-256 per platform. AMADEUS
deliberately ignores any `ffmpeg` found on `PATH`, so an analysis video produced
on one machine is produced by the same encoder on every other machine. The
version actually used is recorded in each conversion's metadata file.

`imageio-ffmpeg` 0.6.0 contains one FFmpeg binary per platform wheel:

| Platform wheel | Bundled binary |
| --- | --- |
| `manylinux2014_x86_64` | `ffmpeg-linux-x86_64-v7.0.2` |
| `manylinux2014_aarch64` | `ffmpeg-linux-aarch64-v7.0.2` |
| `macosx_11_0_arm64` | `ffmpeg-macos-aarch64-v7.1` |
| `macosx_10_9_x86_64` | `ffmpeg-macos-x86_64-v7.1` |
| `win_amd64` | `ffmpeg-win-x86_64-v7.1.exe` |
| `win32` | `ffmpeg-win32-v4.2.2.exe` (not used; AMADEUS requires 64-bit Python) |

`imageio-ffmpeg` is licensed under the BSD 2-Clause License.

Copyright (c) 2018-2025, imageio contributors. All rights reserved.

FFmpeg itself is a separate program. It is licensed under the GNU Lesser General
Public License version 2.1 or later; the builds distributed with
`imageio-ffmpeg` are configured with `--enable-gpl --enable-version3` and
GPL-licensed components (including libx264 and libx265), which places those
binaries under the GNU General Public License version 3 or later. AMADEUS
invokes FFmpeg as an external executable and does not link against it. See
<https://ffmpeg.org/legal.html> for the FFmpeg licensing terms and
<https://github.com/imageio/imageio-ffmpeg> for the binary distribution.

Setting the `AMADEUS_FFMPEG` environment variable makes AMADEUS use the FFmpeg
binary it names instead of the pinned one. That is an explicit, per-site choice;
the substituted version is still written into every conversion's metadata.
