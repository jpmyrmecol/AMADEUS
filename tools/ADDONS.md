# Independent add-ons

## Using add-ons

Select **Add-ons** on Home. Drop one add-on ZIP into the installer, or use **Browse**. Review the developer, license, OS support, update source and SHA-256 before accepting installation. Package inspection does not execute its code. After approval, the chooser closes and an animated green installation bar appears. The progress window closes automatically after installation. The drop area expands to fill the available chooser space.

Installed applications appear in one scrollable column on the right of Home. Panels show the name, version and declared OS/CPU support. Click a panel to launch its independent GUI. Use its **...** menu for **Update** and **Uninstall**. Drag a panel vertically to reorder it. The order is saved between sessions and across updates, and is updated when an add-on is uninstalled. The existing video drop targets continue to open their existing GUIs.

OS support is a developer declaration, not a test certification. An unsupported package cannot be installed on the current OS/CPU. An already registered application remains visible but cannot be launched when its declared platforms do not include the current runtime. GPU requirements and native OS libraries are the add-on developer's responsibility.

An add-on runs in a separate process with its own virtual environment and an external uv-managed Python installation. It does not inherit AMADEUS's Python import path or Tcl/Tk configuration. Starting an installed add-on does not synchronize packages or require an internet connection unless the application itself does so. Initial installation and updates can download Python and dependencies.

Close an add-on before updating or removing it. Home reports failures and leaves the previous installation selected if setup of a replacement fails. An in-progress add-on operation blocks this Home window from closing, updating or uninstalling AMADEUS.

## Persistent storage

| OS | Directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\AMADEUS\addons` |
| macOS | `~/Library/Application Support/AMADEUS/addons` |
| Linux / WSL2 | `$XDG_DATA_HOME/AMADEUS/addons`, or `~/.local/share/AMADEUS/addons` when XDG_DATA_HOME is unset, empty or relative |

On Linux, the AMADEUS application and its add-ons share the `AMADEUS` directory for a standard installation. Existing add-ons in the former lowercase `amadeus/addons` directory are moved to `AMADEUS/addons` on the first launch of the new Home screen. A symbolic link at the former location is retained because installed Python environments can reference absolute paths. This link is removed when AMADEUS is uninstalled. Close running add-ons before migration. If both the old and new add-on directories already contain data, the migration stops and reports the conflict without overwriting either directory. Do not manually delete the compatibility link while the migrated add-ons are installed.

Each add-on has an immutable ID and its own directory:

```text
<id>/
  current.json
  releases/<installation-id>/
    payload/
    .venv/
  data/
  install.log
  run.log
```

A replacement environment is built at its final path before `current.json` is atomically replaced. Existing virtual environments are not moved. Each ID has an OS file lock for management operations. Old program/environment directories are removed after a successful replacement. **Updates preserve `data/`, `run.log`, and previous entries in `install.log` (new installation output is appended).** Failed attempts never become the current installation.

Home reconstructs the panels from these records. Panel ordering is stored in `.order.json` at the add-on storage root. Individual uninstall removes that add-on ID from the saved order. AMADEUS updates, replacement checkouts and reinstalls therefore do not need to copy add-ons out of the application directory. Corrupt records are reported in the installer without preventing Home from opening.

**Uninstalling one add-on** from its panel permanently deletes the complete `<id>/` directory, including the program, virtual environments, `data/` (saved settings and generated outputs), `install.log` and `run.log`. Files outside that add-on directory are not removed. The shared `.python` and `.cache` directories remain available for other add-ons.

**Uninstalling AMADEUS itself** also deletes the entire add-on storage directory, including every add-on, its saved settings, data and logs, and shared add-on Python environments and caches. Videos, tracking sessions and results stored outside the AMADEUS installation and add-on storage are not deleted. Independent applications are not sandboxed and can modify files themselves. Back up important research data.

Do not move installed virtual environments manually. Install the ZIP again at the new location instead. Do not store user data inside `payload/` or `.venv/`; those directories are replaced on update.

## Package format 1

This is an AMADEUS-specific format, not a general Python package standard. A ZIP contains `addon.json`, `LICENSE`, the entry script and application files. One enclosing directory is also accepted. Archives containing Git metadata, virtual environments or Python caches are rejected. The official template's build script produces the intended release ZIP.

```json
{
  "manifest_version": 1,
  "id": "org.example.behavior-analysis",
  "name": "Behavior Analysis",
  "version": "1.0.0",
  "author": "Your name or organization",
  "description": "Describe the application and any additional system requirements.",
  "license": "MIT",
  "platforms": ["windows-x86_64", "macos-arm64", "linux-x86_64"],
  "python": "3.12",
  "entry_point": "app.py",
  "dependencies": ["PyYAML==6.0.2"],
  "drop_extensions": [".csv", ".yaml", ".mp4"],
  "update_url": null
}
```

`id`, `name`, `version`, `author`, `license`, `platforms`, `python`, `entry_point` and `manifest_version` are required. `description`, `dependencies`, `update_url` and `drop_extensions` are optional. Unknown fields and unsupported manifest versions are rejected.

`drop_extensions` is an optional list of distinct lowercase filename suffixes, such as `.csv`, `.yaml`, `.mp4` or `.avi`. Each suffix starts with a dot, followed by 1 to 12 lowercase letters or digits. Up to 32 suffixes are allowed. Omission or an empty list disables drag-and-drop on that add-on's Home panel. If declared, only existing files with those suffixes are accepted, one file at a time. The declaration is opt-in. The add-on must implement the `--dropped-file` argument to process the selected file. Supported suffixes are displayed on its panel.

IDs are lowercase qualified names containing at least one dot or hyphen. Keep the ID unchanged across updates. Versions are numeric `MAJOR.MINOR.PATCH`, with no leading zeros or prerelease suffixes. Reinstalling the current version is allowed. Downgrades are rejected.

Recognized platform values are `windows-x86_64`, `windows-arm64`, `macos-arm64`, `macos-x86_64`, `linux-x86_64` and `linux-arm64`. List only platforms the application is intended to support, and document which were actually tested.

`python` is `3.12`, `3.13` or `3.14`. `entry_point` is a relative `.py` file inside the package. Version 1 supports Python applications, not arbitrary executable installers. `dependencies` contains exact package pins such as `numpy==1.26.4`. Extras are accepted, for example `package[extra]==1.2.3`. Package source URLs, custom indexes, platform markers and source builds are not part of format 1. Dependencies must have compatible wheels on PyPI. Their transitive dependencies are resolved by uv; exact direct pins alone are not a complete dependency lock.

The installer uses the uv executable prepared by the existing AMADEUS launcher. It creates a separate runtime outside the AMADEUS checkout. It never installs add-on packages into AMADEUS's `.venv`.

## File interface

Home normally launches the entry script with no input arguments. The application supplies its own file selectors for AMADEUS results and `config.yaml`. When a supported file is dropped on an opt-in Home panel, AMADEUS launches its independent entry script with `--dropped-file <absolute-path>`. Video files and other inputs can use the same protocol. AMADEUS passes the path unchanged and leaves interpretation or playback to the add-on. Drop support is disabled for add-ons that do not declare `drop_extensions` or are unsupported on the current OS/CPU.

The launcher also supports optional `--config <absolute-path>` and `--results <absolute-path>` arguments through `Store.launch`. Applications following this protocol should accept these options. No AMADEUS Python object is passed across the boundary. `AMADEUS_ADDON_DATA_DIR` supplies the persistent per-application data directory. The process working directory is this data directory, so application resources should be located relative to `__file__`, not the working directory.

The template reads CSV and YAML files without interpreting coordinates or modifying scientific values. Its CSV record count excludes only the first CSV row and is not a frame count or animal count. Developers must handle the actual AMADEUS result schema and configuration version required for their analysis.

## Updates

With `update_url: null`, **Update** opens a ZIP selector. This works for private distributions and manually downloaded packages. The selected ZIP must have the same add-on ID. The Add-ons installer can also install a newer ZIP for an existing ID.

For network updates, `update_url` is a stable HTTPS URL serving a small JSON document:

```json
{
  "manifest_version": 1,
  "id": "org.example.behavior-analysis",
  "version": "1.1.0",
  "url": "https://example.org/releases/behavior-analysis-1.1.0.zip",
  "sha256": "REPLACE_WITH_THE_64_CHARACTER_LOWERCASE_SHA256_OF_THE_ZIP"
}
```

The template's build script generates the real digest. Publish the ZIP first, then publish the update document at the stable URL. The descriptor and archive must be accessible without authentication. Do not place access tokens in URLs. The host verifies the package ID, version and digest, and requests consent again before installation. The confirmation includes the previous and proposed update sources.

There are no automatic background network update checks for add-ons. Updating AMADEUS does not automatically update any add-on. A private GitHub repository is not readable anonymously by the installed application, even when ChatGPT or the developer has GitHub access. Use local ZIP updates for private packages.

## Security and licensing boundary

Add-ons are executable third-party software with the user's file and network permissions. Separate processes and virtual environments are not security sandboxes. Developer names and license labels are self-declared. An HTTPS URL and matching digest establish transport/integrity checks, not an independent verification of the publisher.

Before installation the manager rejects path traversal, absolute paths, Windows reserved paths, duplicate/case-colliding entries, links, special files and encrypted entries. Limits are 512 MiB compressed, 1 GiB expanded, 10000 entries and 64 KiB of JSON metadata. No package installation hooks are imported by Home. Dependencies are installed from wheels after user approval. Never treat these checks as a guarantee that the application is safe.

The AMADEUS management implementation remains AGPL-3.0-only. The [official template](https://github.com/jpmyrmecol/AMADEUS-Addon-Template) is independently written under 0BSD. An add-on should not copy or import AMADEUS implementation modules. It communicates using files and runs independently. This architecture does not change the licenses of AMADEUS, Ultralytics, or any dependency. Assess the licenses of all code included in each add-on.

## References

[uv CLI](https://docs.astral.sh/uv/reference/cli/), [Python virtual environments](https://docs.python.org/3/library/venv.html), [ZIP safety](https://docs.python.org/3/library/zipfile.html), [0BSD](https://spdx.org/licenses/0BSD.html).
