# AMADEUS

**Annotation-free Multi-Animal Direction Estimation Using Self-supervised learning**

AMADEUS is a markerless multi-animal tracking system for laboratory videos. It estimates an oriented bounding box (OBB) and head direction for each animal while maintaining individual identities over time. No manual training annotation or physical marking is required.

After you configure foreground segmentation, AMADEUS extracts single-animal blobs, assigns direction classes using movement direction, and synthesizes interaction images by copy-paste augmentation. A detector trained on these images estimates OBBs and direction classes during crossings and crowding. Staged association and refinement produce individual tracks. When almost complete occlusion is expected, contrastive learning is additionally used for identity verification and correction.

- [Website](https://amadeus.jpmyrmecol.com/)
- [Online manual](https://amadeus.jpmyrmecol.com/Manual_EN.html)

## Install and launch

Use the [installers on the website](https://amadeus.jpmyrmecol.com/#install), or clone this repository and run the launcher from its root directory.

| Operating system | Launcher |
| --- | --- |
| Windows 10 / 11 | Double-click `AMADEUS.bat` |
| macOS on Apple Silicon | Open `AMADEUS.command` |
| Linux / WSL2 | Run `bash AMADEUS.sh` |

macOS and Linux have not yet been extensively tested. For platform requirements, installation details, and troubleshooting, see the [manual](https://amadeus.jpmyrmecol.com/Manual_EN.html).

The launcher prepares a locked Python environment with [uv](https://docs.astral.sh/uv/). Because uv resolves `uv.lock`, its version is pinned too: the version in `UV_VERSION` is checked at startup, and if it is not already present AMADEUS installs that exact version into the AMADEUS folder (`.uv/`), leaving any uv you installed yourself untouched.

## Run tracking

1. Open **Easy Tracking** and select the video and session folder.
2. Select **Launch Segmentation**. Configure foreground segmentation so that complete, isolated animals are retained as single-animal blobs and unsuitable regions are marked as outliers.
3. Specify the number of animals, overlap severity, and whether the animals move backward. Select **Processing** to run the workflow.
4. Review the result video and CSV. Use **Refinement** to correct remaining errors if needed.

Use **Advanced Tracking** to configure individual stages and parameters, or **Multi Config Batch** to run several saved configurations.

## Input videos

MP4, MOV, AVI, MTS, M2TS, MKV, MPG/MPEG, TS and WebM can be selected. Whether a file can be used is decided by inspecting it, not by its extension: AMADEUS reads its codec, pixel format, frame rate mode, colour transfer and rotation with FFmpeg, then checks that OpenCV can decode it and seek to an arbitrary frame, which is how segmentation and tracking read frames.

A video that already passes -- an 8-bit H.264 MP4/MOV/AVI with a constant frame rate, as before -- is used exactly as it is, with no prompt and no re-encoding. Anything else (HEVC, 10-bit, HDR, variable frame rate, rotation metadata, transport-stream containers) raises a dialog that explains what was found and offers to write an **analysis copy** beside the video, in `amadeus_<name>/converted/`. Conversion never starts on its own, and the original file is never modified.

The analysis copy keeps the original resolution and is normalised to H.264 MP4, 8-bit SDR, constant frame rate, CRF 12 with short keyframe intervals so frame seeking stays exact. HDR is tone mapped to SDR, and rotation metadata is applied to the pixels. A frame range can be chosen; it defaults to the whole video. Each copy is written with a `.amadeus_conversion.json` record naming the source, the frame range, the frame rate, the codec, the FFmpeg version and how a converted frame number maps back onto the source, so a result stays reproducible.

AMADEUS uses the FFmpeg build pinned by `imageio-ffmpeg` to inspect and convert analysis videos. On Windows and Linux, the launcher prepares a separate checksum-pinned GPU-capable FFmpeg build when it detects a GPU; existing installs prepare it when Cropping & Trimming is first opened. NVIDIA NVENC, Intel Quick Sync, and AMD AMF are tested before the hardware encoding option is enabled. The final H.264 encoding uses the GPU; crop, rotation, and image adjustments still run on the CPU. If no hardware encoder works, exports use CPU encoding. See [third-party notices](THIRD_PARTY_NOTICES.md).

## Outputs and recording conditions

The standard output includes individual identity, OBB center and dimensions, and head direction in degrees over time. Front and rear keypoints are derived geometrically from OBBs. In the final CSV, columns named `heading` store head direction. See the [output format](https://amadeus.jpmyrmecol.com/Manual_EN.html#section-12) for coordinates, angles, and processing stages.

AMADEUS learns directly from the input videos. Recommended conditions include a fixed camera viewing animals from above, movement approximately confined to a two-dimensional plane, stable illumination, an essentially static background, and sufficient contrast, resolution, and frame rate for reliable visual tracking in each frame. Head direction estimation requires a body axis and visually distinguishable front and rear. The video must contain sufficient periods in which animals are separated from one another to generate training data. Training and analysis videos can be recorded separately under the same conditions.

**Variable population** supports animals entering or leaving the field of view, but an animal may receive a new ID after returning. **Without direction estimation** tracks OBBs and identities without assigning front or rear, allowing AMADEUS to be applied to animals without a visually distinguishable front and rear. See the [manual](https://amadeus.jpmyrmecol.com/Manual_EN.html#section-6) for these modes and their limitations.

## Citation

See the [Publication page](https://amadeus.jpmyrmecol.com/publication.html) for citation information when the preprint is available. Please also cite the project website: https://amadeus.jpmyrmecol.com/.

## License

AMADEUS is distributed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only). See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
