# Independent add-ons

## Using add-ons

Select **Add-ons** on Home. Drop one add-on ZIP into the installer, or use **Browse**. Review the developer, license, OS support, update source and SHA-256 before accepting installation. Package inspection does not execute its code.

Installed applications appear in one scrollable column on the right of Home. Panels show the name, version and declared OS/CPU support. Click a panel to launch its independent GUI. Use its **...** menu for **Update** and **Uninstall**. The existing video drop targets continue to open their existing GUIs.

OS support is a developer declaration, not a test certification. An unsupported package cannot be installed on the current OS/CPU. An already registered application remains visible but cannot be launched when its declared platforms do not include the current runtime. GPU requirements and native OS libraries are the add-on developer's responsibility.

An add-on runs in a separate process with its own virtual environment and an external uv-managed Python installation. It does not inherit AMADEUS's Python import path or Tcl/Tk configuration. Starting an installed add-on does not synchronize packages or require an internet connection unless the application itself does so. Initial installation and updates can download Python and dependencies.

Close an add-on before updating or removing it. Home reports failures and leaves the previous installation selected if setup of a replacement fails. An in-progress add-on operation blocks this Home window from closing, updating or uninstalling AMADEUS.

## Persistent storage

| OS | Directory |
| --- | --- |
| Windows | `%LOCALAPPDATA%\AMADEUS\addons` |
| macOS | `~/Library/Application Support/AMADEUS/addons` |
| Linux | `$XDG_DATA_HOME/amadeus/addons`, or `~/.local/share/amadeus/addons` when XDG_DATA_HOME is not an absolute path |

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

A replacement environment is built at its final path before `current.json` is atomically replaced. Existing virtual environments are not moved. Each ID has an OS file lock for management operations. Old program/environment directories are removed after a successful replacement. Failed attempts never become the current installation.

Home reconstructs the panels from these records. AMADEUS updates, replacement checkouts and reinstalls therefore do not need to copy add-ons out of the application directory. Corrupt records are reported in the installer without preventing Home from opening.

**Uninstall removes the add-on program and environments but retains `data/` and logs.** The manager does not remove input videos, result CSV files or configuration files. The shared `.python` and `.cache` directories also remain. Independent applications are not sandboxed and can modify files themselves. Back up important research data.

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
  "update_url": null
}
```

`id`, `name`, `version`, `author`, `license`, `platforms`, `python`, `entry_point` and `manifest_version` are required. `description`, `dependencies` and `update_url` are optional. Unknown fields and unsupported manifest versions are rejected.

IDs are lowercase qualified names containing at least one dot or hyphen. Keep the ID unchanged across updates. Versions are numeric `MAJOR.MINOR.PATCH`, with no leading zeros or prerelease suffixes. Reinstalling the current version is allowed. Downgrades are rejected.

Recognized platform values are `windows-x86_64`, `windows-arm64`, `macos-arm64`, `macos-x86_64`, `linux-x86_64` and `linux-arm64`. List only platforms the application is intended to support, and document which were actually tested.

`python` is `3.12`, `3.13` or `3.14`. `entry_point` is a relative `.py` file inside the package. Version 1 supports Python applications, not arbitrary executable installers. `dependencies` contains exact package pins such as `numpy==1.26.4`. Extras are accepted, for example `package[extra]==1.2.3`. Package source URLs, custom indexes, platform markers and source builds are not part of format 1. Dependencies must have compatible wheels on PyPI. Their transitive dependencies are resolved by uv; exact direct pins alone are not a complete dependency lock.

The installer uses the uv executable prepared by the existing AMADEUS launcher. It creates a separate runtime outside the AMADEUS checkout. It never installs add-on packages into AMADEUS's `.venv`.

## File interface

Home normally launches the entry script with no input arguments. The application supplies its own file selectors for AMADEUS results and `config.yaml`.

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
