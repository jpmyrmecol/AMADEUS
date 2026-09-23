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

AMADEUS uses two FFmpeg builds for different tasks.

### Inspecting and converting analysis videos

The FFmpeg build is pinned, not discovered on the host: it is the binary shipped
inside the `imageio-ffmpeg` wheel fixed at `imageio-ffmpeg==0.6.0` in
`pyproject.toml` and pinned by SHA-256 per platform in `uv.lock`. AMADEUS
ignores `ffmpeg` on `PATH` for analysis-video inspection and conversion, so
those operations use the same FFmpeg build across machines. The version used
for each conversion is recorded in its metadata file.

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
<https://ffmpeg.org/legal.html> for FFmpeg licensing terms and
<https://github.com/imageio/imageio-ffmpeg> for the binary distribution.

Setting the `AMADEUS_FFMPEG` environment variable makes AMADEUS use the FFmpeg
binary it names instead of the pinned one. That is an explicit, per-site choice;
the substituted version is still written into every conversion's metadata.

### Cropping & Trimming hardware encoding

Cropping & Trimming probes the pinned FFmpeg for a usable hardware encoder. If
the host exposes a GPU and that FFmpeg cannot encode with it, AMADEUS downloads
a separate FFmpeg build into `.ffmpeg-hardware/` on supported Windows and Linux
systems. The archive URL, size, and SHA-256 are pinned in
`tools/hardware_ffmpeg.py`; the archive is verified before the FFmpeg binary
and its `LICENSE.txt` are installed. The source FFmpeg build is
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds), GPL static variant,
build `N-126342` from 2026-08-31. Its build configuration includes NVIDIA NVENC, Intel Quick Sync where
supported, AMD AMF, and VAAPI on Linux x86_64. The upstream Linux ARM64 build does not
include Intel QSV or VAAPI.

The downloaded build is licensed under GPL version 3 or later because the GPL
variant enables GPL-licensed FFmpeg components. AMADEUS invokes it as an
external executable and does not link against it. The upstream build repository
contains the corresponding build scripts and dependency notices; FFmpeg source
and license information are at <https://ffmpeg.org/legal.html>. The archive's
`LICENSE.txt` is kept beside the installed binary.

Compilation support does not guarantee that a particular encoder will work with
every GPU or driver. AMADEUS enables the hardware-encoding option only after a
short encode succeeds with the detected FFmpeg and driver. On macOS, it tests
the pinned build for Apple VideoToolbox; the downloaded BtbN builds currently
cover Windows and Linux. Hardware encoding applies to the final H.264 encoding
step; cropping, rotation, and image adjustments are still performed on the CPU.
