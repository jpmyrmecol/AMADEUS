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

AMADEUS uses one fixed FFmpeg executable per operating system for video
inspection, analysis conversion, Cropping & Trimming, and Create Video. It does
not select FFmpeg from `PATH` or keep a second GPU-only binary.

On Windows and Linux, the launcher downloads one checksum-pinned BtbN static
build into `.ffmpeg-hardware/` during setup. The same binary and its `LICENSE.txt`
are used by all video features. The pinned assets, archive sizes, and SHA-256
checks are in `tools/ffmpeg_runtime.py`; the source is
[BtbN/FFmpeg-Builds](https://github.com/BtbN/FFmpeg-Builds), GPL static variant,
build `N-126342` from 2026-08-31. Separate archives cover Windows and Linux on
x86_64 and ARM64. The GPL builds include the `libx264` CPU encoder. GPU encoder
support depends on the target architecture. In particular, these pinned builds
disable Intel Quick Sync on ARM64 and VAAPI on Linux ARM64. BtbN's static binaries target Windows 10 22H2
or newer and Linux glibc 2.28 / kernel 4.18 or newer.

On macOS, AMADEUS uses the platform-specific `imageio-ffmpeg==0.6.0` binary,
which includes Apple VideoToolbox support. This dependency is installed only on
macOS; Windows and Linux do not install an additional `imageio-ffmpeg` binary.
The `imageio-ffmpeg` Python package is licensed under the BSD 2-Clause License.
The bundled FFmpeg executable has its own FFmpeg and codec licensing notices.

Copyright (c) 2018-2025, imageio contributors. All rights reserved.

The BtbN FFmpeg build is licensed under GPL version 3 or later because the GPL
variant enables GPL-licensed FFmpeg components. AMADEUS invokes it as an
external executable and does not link against it. The upstream build repository
contains corresponding build scripts and dependency notices. FFmpeg source and
license information are at <https://ffmpeg.org/legal.html>; the downloaded
archive's `LICENSE.txt` is kept beside the installed binary. The macOS
`imageio-ffmpeg` package is licensed under the BSD 2-Clause License.

Compilation support does not guarantee that a particular encoder will work with
every GPU or driver. AMADEUS selects candidates from detected GPUs: NVIDIA NVENC
on Windows and Linux, AMD AMF or Intel Quick Sync on Windows, VAAPI for AMD/Intel
on Linux, and Apple VideoToolbox on macOS. It only tests these candidates with
the pinned FFmpeg executable and the options used by the requested export path.
VAAPI tests and exports use the same detected render device. VideoToolbox must
use its hardware implementation. Results are cached per executable and export
profile for the process lifetime.

GPU acceleration is enabled by default in Cropping & Trimming only after a
short encode succeeds. If no candidate succeeds, GPU acceleration is disabled
and H.264 exports use CPU `libx264`. Create Video uses the same validation with
and without direction estimation. `ACCELERATION=cpu` always uses CPU encoding.
`auto` selects a validated GPU encoder or falls back to CPU. `gpu` retains the
same fallback with a warning when GPU acceleration is unavailable. An explicit
`VIDEO_CODEC=mp4v` selection retains CPU MPEG-4 encoding.

GPU acceleration applies to final H.264 encoding. Cropping, rotation, image
adjustments, and overlay rendering still run on the CPU. A short test cannot
guarantee support for every resolution, concurrent session, or later driver
state. Errors during an actual export are reported without silently restarting
the entire export on the CPU. GPU test failures do not prevent application
startup. The pinned FFmpeg inventory requires `libx264`, independently of GPU
encoder support.
