# oklab-krita-colour-picker

The OKLab/OKCLh colour space was introduced in 2020 [by Björn Ottosson](https://bottosson.github.io/posts/oklab/).

This repo contains an attempt to bring OKLab/OKCLh colour picker to Krita in performant and user-friendly format that are aligned with the Krita's software phylosophy.

More information to come. Stay tuned...

## Installing from source

The plugin lives in `oklab_colour_picker/`, with the Krita manifest at
`oklab_colour_picker.desktop`.

### Prerequisites

- Krita 5.2 or newer (Python plugin support, PyQt5).
- **NumPy** available to Krita's Python. Krita's bundled Python on Windows and
  macOS does not ship NumPy. If NumPy is missing the docker will load with a
  friendly placeholder message instead of the colour selector.

Installing NumPy into Krita's Python:

- **Linux** — Krita typically uses the system Python. Install via your package
  manager (`sudo apt install python3-numpy`, `sudo dnf install python3-numpy`,
  etc.) or `pip install --user numpy`.
- **Windows** — if NumPy is missing, open the docker and click
  **Install NumPy**. The plugin uses Krita's bundled `python.exe` to download
  NumPy from PyPI into `oklab_colour_picker/site-packages` under Krita's app
  data folder, then prompts you to restart Krita. The plugin adds that folder
  to `sys.path` before importing NumPy. To install manually instead, run from
  a Command Prompt:
  ```
  "C:\Program Files\Krita (x64)\bin\python.exe" -m pip install numpy
  ```
- **macOS** — Krita's bundled Python lives inside the app bundle. From a
  Terminal:
  ```sh
  /Applications/krita.app/Contents/MacOS/krita_python -m pip install numpy
  ```
  If `krita_python` is not present on your build, follow the Krita docs for
  installing Python packages on your platform.

### 1. Locate Krita's resource folder

In Krita: **Settings → Manage Resources… → Open Resource Folder**. Typical
defaults:

- Linux: `~/.local/share/krita/`
- macOS: `~/Library/Application Support/krita/`
- Windows: `%APPDATA%\krita\`

If a `pykrita/` subfolder does not exist, create one.

### 2. Copy the plugin into `pykrita/`

From a clone of this repo, copy the package folder and the `.desktop` manifest
into `pykrita/` so you end up with:

```
pykrita/
├── oklab_colour_picker.desktop
└── oklab_colour_picker/
    ├── __init__.py
    ├── plugin.py
    └── ... (the rest of the package)
```

Linux:

```sh
cp -r oklab_colour_picker          ~/.local/share/krita/pykrita/
cp    oklab_colour_picker.desktop  ~/.local/share/krita/pykrita/
```

macOS:

```sh
cp -r oklab_colour_picker          ~/Library/Application\ Support/krita/pykrita/
cp    oklab_colour_picker.desktop  ~/Library/Application\ Support/krita/pykrita/
```

Windows (Command Prompt):

```
xcopy /E /I oklab_colour_picker "%APPDATA%\krita\pykrita\oklab_colour_picker"
copy oklab_colour_picker.desktop "%APPDATA%\krita\pykrita\"
```

You may symlink instead of copying if you want `git pull` to update the
installed version.

### 3. Enable the plugin

1. Restart Krita.
2. **Settings → Configure Krita… → Python Plugin Manager**.
3. Tick **OKLab Colour Selector** and click **OK**.
4. Restart Krita again so the plugin is loaded.

### 4. Open the docker

**Settings → Dockers → OKLab Colour Selector**. Drag it into your workspace
like any other docker.

### Troubleshooting

- **Plugin missing from the Plugin Manager** — confirm
  `oklab_colour_picker.desktop` sits directly inside `pykrita/`, not nested
  inside the `oklab_colour_picker/` subfolder.
- **Docker shows "missing dependency" message** — click **Install NumPy** in
  the docker, or install NumPy manually (see Prerequisites), then restart Krita.
- **Errors on startup** — open **Tools → Scripts → Python Script Editor**, or
  launch Krita from a terminal, to see the traceback.
- **Edits don't appear** — Krita reloads Python plugins only on restart.
