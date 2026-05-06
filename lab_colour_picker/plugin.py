"""Krita plugin registration for the OKLab colour picker docker."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


DOCK_FACTORY_ID = "oklab_colour_picker_dock"
DOCK_TITLE = "OKLab Colour Selector"
DOCK_AREA_NAME = "DockRight"


@dataclass(frozen=True)
class KritaApi:
    Krita: object
    DockWidget: type
    DockWidgetFactory: type
    DockWidgetFactoryBase: object


def register_plugin(*, krita_instance=None, api: KritaApi | None = None) -> bool:
    krita_api = api if api is not None else _load_krita_api()
    if krita_api is None:
        return False

    app = krita_instance if krita_instance is not None else krita_api.Krita.instance()
    dock_class = create_dock_widget_class(krita_api.DockWidget)
    dock_area = getattr(krita_api.DockWidgetFactoryBase, DOCK_AREA_NAME)
    factory = krita_api.DockWidgetFactory(DOCK_FACTORY_ID, dock_area, dock_class)
    app.addDockWidgetFactory(factory)
    return True


def create_dock_widget_class(dock_widget_base: type, *, controller_factory: Callable | None = None) -> type:
    class OKLabColourPickerDock(dock_widget_base):
        def __init__(self) -> None:
            super().__init__()
            from lab_colour_picker.dock import ColourPickerDockPanel, connect_dock_visibility

            self._controller = _create_controller() if controller_factory is None else controller_factory()
            self._panel = ColourPickerDockPanel(self._controller, self)
            self.setWindowTitle(DOCK_TITLE)
            self.setWidget(self._panel)
            self._visibility_connection = connect_dock_visibility(self, self._controller)
            self.destroyed.connect(self._disconnect_visibility)

        def canvasChanged(self, canvas) -> None:
            pass

        def _disconnect_visibility(self) -> None:
            self._visibility_connection.disconnect()

    OKLabColourPickerDock.__name__ = "OKLabColourPickerDock"
    return OKLabColourPickerDock


def _create_controller():
    from lab_colour_picker.controller import ColourPickerController
    from lab_colour_picker.krita_adapter import KritaForegroundAdapter, QtForegroundTimer, QtSingleShotScheduler

    return ColourPickerController(
        KritaForegroundAdapter(),
        scheduler=QtSingleShotScheduler(),
        foreground_timer=QtForegroundTimer(),
    )


def _load_krita_api() -> KritaApi | None:
    try:
        from krita import DockWidget, DockWidgetFactory, DockWidgetFactoryBase, Krita
    except ImportError:
        return None
    return KritaApi(
        Krita=Krita,
        DockWidget=DockWidget,
        DockWidgetFactory=DockWidgetFactory,
        DockWidgetFactoryBase=DockWidgetFactoryBase,
    )
