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

## PyAV (reverse playback)

Refinement uses the pinned `av==16.0.1` (PyAV) package for access to
FFmpeg video decoding and for retaining decoded frames in its reverse-playback
buffer. AMADEUS installs PyAV as a Python dependency; PyAV binary wheels
are not checked into this repository.

PyAV's own source code is licensed under the BSD 3-Clause License.
The following copyright and license notice is reproduced from
[PyAV v16.0.1 LICENSE.txt](https://github.com/PyAV-Org/PyAV/blob/v16.0.1/LICENSE.txt):

### PyAV BSD 3-Clause License

Copyright retained by original committers. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the project nor the names of its contributors may be
      used to endorse or promote products derived from this software without
      specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

### FFmpeg and codecs in PyAV wheels

The official PyAV binary wheels include FFmpeg libraries and other codec
dependencies built for PyAV. These are **separate** from the FFmpeg executable
that AMADEUS manages for video inspection and export, described below.

The BSD 3-Clause License above covers PyAV's own code; it does not replace
the licenses of the FFmpeg libraries or other components in those wheels.
FFmpeg is generally LGPL-2.1-or-later, but an FFmpeg build that enables
GPL-covered components is subject to GPL terms. Other bundled libraries may
have their own license and source-distribution requirements. The applicable
terms depend on the precise wheel and platform-specific build configuration;
this notice does not assert a single license for every PyAV wheel.

When redistributing PyAV wheels or packaging them into a standalone AMADEUS
distribution, review the corresponding build's license notices, component
licenses, and any applicable source-availability obligations. References:

- [PyAV v16.0.1 source and license](https://github.com/PyAV-Org/PyAV/tree/v16.0.1)
- [PyAV FFmpeg binary build sources](https://github.com/PyAV-Org/pyav-ffmpeg)
- [FFmpeg licensing and compliance guidance](https://ffmpeg.org/legal.html)

## FFmpeg

AMADEUS uses one fixed FFmpeg executable per operating system for video
inspection, analysis conversion, Cropping & Trimming, Create Video, and
Refinement video export. It does
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

On Windows and Linux, Cropping & Trimming enables GPU encoding by default
only after a short encode succeeds. On macOS its default is CPU `libx264`
with CRF 18, while the VideoToolbox checkbox offers faster hardware encoding.
Saved GPU selections remain effective. A failed GPU probe disables hardware
encoding, and H.264 exports use CPU `libx264`.

Create Video uses the same selection policy in every model mode and without
direction estimation. `ACCELERATION=cpu` always uses CPU encoding, and
`ACCELERATION=auto` uses CPU `libx264` on macOS but prefers validated
hardware encoding on Windows and Linux. `ACCELERATION=gpu` explicitly
requests hardware encoding, including VideoToolbox on macOS, and falls back
to CPU with a warning when hardware encoding is unavailable. An explicit
`VIDEO_CODEC=mp4v` selection retains CPU MPEG-4 encoding.

Video exports use high-quality encoder settings rather than limits based on
the source bitrate. CPU H.264 uses x264 CRF 18; hardware options are NVENC
CQ 18, QSV/AMF/VAAPI quality 18 and VideoToolbox quality 85. These numbers
do not correspond to identical visual quality across encoders. Output can be
larger than its source when colored annotations add detail. Refinement uses
CPU H.264 for MP4 and AVI. Tracking videos are encoded directly from rendered
frames, including when separate JPEG or PNG images are exported.

GPU acceleration applies to final H.264 encoding. Cropping, rotation, image
adjustments, and overlay rendering still run on the CPU. A short test cannot
guarantee support for every resolution, concurrent session, or later driver
state. Errors during an actual export are reported without silently restarting
the entire export on the CPU. GPU test failures do not prevent application
startup. The pinned FFmpeg inventory requires `libx264`, independently of GPU
encoder support.
