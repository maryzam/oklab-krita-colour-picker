# Troubleshooting

## Plugin Missing From Krita

Confirm both files are directly inside Krita's `pykrita/` folder:

```text
pykrita/
+-- oklab_colour_picker.desktop
+-- oklab_colour_picker/
```

The `.desktop` file should not be nested inside the package folder.

After changing plugin files, restart Krita. Python plugins are loaded at
startup.

## Docker Shows A Missing Dependency Message

NumPy is not available to Krita's Python.

Use the docker's **Install NumPy** button if it appears, or install NumPy
manually using [install.md](install.md). Restart Krita after installation.

## Docker Does Not Open

Check that the plugin is enabled:

1. Open **Settings > Configure Krita... > Python Plugin Manager**.
2. Enable **OKLab Colour Selector**.
3. Restart Krita.
4. Open **Settings > Dockers > OKLab Colour Selector**.

## Startup Errors

Open **Tools > Scripts > Python Script Editor** in Krita to inspect Python
errors. You can also launch Krita from a terminal and read the traceback there.

## Local Edits Do Not Appear

Restart Krita after editing plugin files. For active development, use symlinks
from Krita's `pykrita/` folder to this repository so the installed files point
at your working copy.
