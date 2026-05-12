# Install

The plugin lives in `oklab_colour_picker/`, with the Krita manifest at
`oklab_colour_picker.desktop`.

## Requirements

- Krita 5.2 or newer with Python plugin support.
- PyQt5, provided by Krita.
- NumPy available to Krita's Python.

Krita's bundled Python on Windows and macOS may not include NumPy. If NumPy is
missing, the docker shows a dependency message instead of the colour selector.

## Locate Krita's Resource Folder

In Krita, open **Settings > Manage Resources... > Open Resource Folder**.

Typical locations:

- Linux: `~/.local/share/krita/`
- macOS: `~/Library/Application Support/krita/`
- Windows: `%APPDATA%\krita\`

Create a `pykrita/` folder there if it does not already exist.

## Copy The Plugin

Copy the package folder and desktop manifest into `pykrita/`:

```text
pykrita/
+-- oklab_colour_picker.desktop
+-- oklab_colour_picker/
    +-- __init__.py
    +-- plugin.py
    +-- ...
```

Linux:

```sh
cp -r oklab_colour_picker ~/.local/share/krita/pykrita/
cp oklab_colour_picker.desktop ~/.local/share/krita/pykrita/
```

macOS:

```sh
cp -r oklab_colour_picker ~/Library/Application\ Support/krita/pykrita/
cp oklab_colour_picker.desktop ~/Library/Application\ Support/krita/pykrita/
```

Windows Command Prompt:

```bat
xcopy /E /I oklab_colour_picker "%APPDATA%\krita\pykrita\oklab_colour_picker"
copy oklab_colour_picker.desktop "%APPDATA%\krita\pykrita\"
```

During development, you can symlink the package and desktop file instead of
copying them.

## Install NumPy

Linux usually uses the system Python. Install NumPy with your package manager
or with pip:

```sh
python3 -m pip install --user numpy
```

On Windows, open the docker and click **Install NumPy** if the dependency
message appears. The plugin installs NumPy into:

```text
%APPDATA%\krita\oklab_colour_picker\site-packages
```

Manual Windows install:

```bat
"C:\Program Files\Krita (x64)\bin\python.exe" -m pip install ^
  --upgrade --only-binary=:all: ^
  --target "%APPDATA%\krita\oklab_colour_picker\site-packages" ^
  "numpy>=1.26,<3"
```

macOS app bundle install:

```sh
/Applications/krita.app/Contents/MacOS/krita_python -m pip install numpy
```

If your Krita build does not include `krita_python`, follow Krita's platform
docs for installing Python packages into its bundled Python.

## Enable The Plugin

1. Restart Krita.
2. Open **Settings > Configure Krita... > Python Plugin Manager**.
3. Enable **OKLab Colour Selector**.
4. Click **OK**.
5. Restart Krita again.

Open the docker from **Settings > Dockers > OKLab Colour Selector**.
