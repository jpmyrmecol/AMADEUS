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

## macOS setup

On Apple Silicon Macs, Python does not need to be installed in advance. Both the website's `AMADEUS-Setup.command` and this repository's `AMADEUS.command` check for a working native Python 3.10–3.12 with Tcl/Tk 8.6 before preparing dependencies. Compatible existing installations are reused, including python.org, Homebrew, and executables on PATH. Intel Macs and Rosetta are not supported.

If none is found, a macOS confirmation dialog offers to download the official python.org package. Choose **Download Python** to download, verify, and open it in **Installer.app**. Choose **Cancel** to stop without downloading. Complete installation and any administrator authentication in Installer.app, then **run AMADEUS Setup again** (or rerun `AMADEUS.command` for a cloned repository). AMADEUS exits after opening the package; it does not run a privileged installer, wait for installation, or resume setup automatically. Download, verification, or opening failures stop setup; rerun the launcher to retry.

The package installs a system-wide Python framework and `/Applications/Python 3.12`; it may replace an existing python.org 3.12 installation. AMADEUS does not remove this shared Python when its own environment is removed. The download is checked against a pinned SHA-256 and macOS package-signature/Gatekeeper checks before opening Installer.app. The temporary package is retained after a successful handoff so Installer.app can read it; it can be deleted after installation. Failed downloads, verification, or opening attempts are cleaned up.

Advanced configuration: set `AMADEUS_PYTHON` to a compatible Python executable or `AMADEUS_VENV` to the environment directory. An invalid explicit Python or an incompatible/incomplete existing environment stops setup without replacement. To preserve an old environment and start fresh, choose a new directory, for example:

```bash
AMADEUS_VENV="$HOME/Applications/AMADEUS/.venv-tk86" bash AMADEUS.command
```

### Bootstrap maintenance

The installer pins [Python 3.12.10](https://www.python.org/downloads/release/python-31210/), the last Python 3.12 release with an official macOS binary installer, for the current Python/Tk compatibility requirements. This is not the latest security-only Python 3.12 release. Review this pin and its security tradeoff when updating GUI dependencies; do not replace it with an untested latest Python/Tk version. The SHA-256 comes from the release's `.pkg.sigstore` message digest. On the next launch, the same native architecture, Python version, Tcl/Tk version, and GUI initialization checks run before environment setup.

`tools/macos_python.sh` is embedded verbatim between the shared-bootstrap markers in `AMADEUS-site/AMADEUS-Setup.command`; update both copies together. This allows the standalone installer to check Python before downloading AMADEUS or uv. Publish the main repository changes before the site changes. Run `python -m unittest discover -s tests` for mocked bootstrap regression tests; native Apple Silicon confirmation dialogs, Installer.app handoff, and GUI behavior require a macOS smoke test.

## Run tracking

1. Open **Easy Tracking** and select the video and session folder.
2. Select **Launch Segmentation**. Configure foreground segmentation so that complete, isolated animals are retained as single-animal blobs and unsuitable regions are marked as outliers.
3. Specify the number of animals, overlap severity, and whether the animals move backward. Select **Processing** to run the workflow.
4. Review the result video and CSV. Use **Refinement** to correct remaining errors if needed.

Use **Advanced Tracking** to configure individual stages and parameters, or **Multi Config Batch** to run several saved configurations.

## Outputs and recording conditions

The standard output includes individual identity, OBB center and dimensions, and head direction in degrees over time. Front and rear keypoints are derived geometrically from OBBs. In the final CSV, columns named `heading` store head direction. See the [output format](https://amadeus.jpmyrmecol.com/Manual_EN.html#section-12) for coordinates, angles, and processing stages.

AMADEUS learns directly from the input videos. Recommended conditions include a fixed camera viewing animals from above, movement approximately confined to a two-dimensional plane, stable illumination, an essentially static background, and sufficient contrast, resolution, and frame rate for reliable visual tracking in each frame. Head direction estimation requires a body axis and visually distinguishable front and rear. The video must contain sufficient periods in which animals are separated from one another to generate training data. Training and analysis videos can be recorded separately under the same conditions.

**Variable population** supports animals entering or leaving the field of view, but an animal may receive a new ID after returning. **Without direction estimation** tracks OBBs and identities without assigning front or rear, allowing AMADEUS to be applied to animals without a visually distinguishable front and rear. See the [manual](https://amadeus.jpmyrmecol.com/Manual_EN.html#section-6) for these modes and their limitations.

## Citation

See the [Publication page](https://amadeus.jpmyrmecol.com/publication.html) for citation information when the preprint is available. Please also cite the project website: https://amadeus.jpmyrmecol.com/.

## License

AMADEUS is distributed under the GNU Affero General Public License v3.0 only (AGPL-3.0-only). See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md).
