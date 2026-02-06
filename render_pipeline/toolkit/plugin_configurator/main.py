# This tool offers an interface to configure the pipeline

from __future__ import print_function

import os
import sys
import time
from threading import Thread
from functools import partial

# Change to the current directory
os.chdir(os.path.join(os.path.dirname(os.path.realpath(__file__))))

sys.path.insert(0, os.getcwd())
sys.path.insert(0, "../../")

from rplibs.six import iteritems  # noqa
from rplibs.pyqt_imports import *  # noqa

from ui.main_window_generated import Ui_MainWindow  # noqa

from rpcore.pluginbase.manager import PluginManager  # noqa
from rpcore.util.network_communication import NetworkCommunication  # noqa
from rpcore.mount_manager import MountManager  # noqa


# =========================================================
# Fixed-point helpers (prevents ALL float→Qt type errors)
# =========================================================

FLOAT_SCALE = 100000


def to_slider(v):
    return int(v * FLOAT_SCALE)


def from_slider(v):
    return v / FLOAT_SCALE


# =========================================================


class PluginConfigurator(QMainWindow, Ui_MainWindow):

    """ Interface to change the plugin settings """

    def __init__(self):

        self._mount_mgr = MountManager(None)
        self._mount_mgr.mount()

        self._plugin_mgr = PluginManager(None)
        self._plugin_mgr.requires_daytime_settings = False

        QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)

        self._current_plugin = None
        self._current_plugin_instance = None

        self.lbl_restart_pipeline.hide()
        self._set_settings_visible(False)
        self._update_queue = []

        qt_connect(self.lst_plugins, "itemSelectionChanged()", self.on_plugin_selected)
        qt_connect(self.lst_plugins, "itemChanged(QListWidgetItem*)", self.on_plugin_state_changed)
        qt_connect(self.btn_reset_plugin_settings, "clicked()", self.on_reset_plugin_settings)

        self._load_plugin_list()

        self.table_plugin_settings.setColumnWidth(0, 140)
        self.table_plugin_settings.setColumnWidth(1, 105)
        self.table_plugin_settings.setColumnWidth(2, 160)

        update_thread = Thread(target=self.update_thread)
        update_thread.daemon = True
        update_thread.start()

    # =========================================================

    def closeEvent(self, event):  # noqa
        event.accept()
        os._exit(1)

    # =========================================================

    def update_thread(self):
        while True:
            if self._update_queue:
                item = self._update_queue.pop(-1)
                NetworkCommunication.send_async(NetworkCommunication.CONFIG_PORT, item)

                if item.startswith("setval "):
                    setting_id = item.split()[1]
                    self._update_queue = [
                        e for e in self._update_queue
                        if e.split()[1] != setting_id
                    ]

            time.sleep(0.2)

    # =========================================================

    def _rewrite_plugin_config(self):
        self._plugin_mgr.save_overrides("/$$rpconfig/plugins.yaml")

    def _show_restart_hint(self):
        self.lbl_restart_pipeline.show()

    # =========================================================

    def _do_update_setting(self, setting_id, value):

        setting_handle = self._plugin_mgr.get_setting_handle(
            self._current_plugin, setting_id)

        if setting_handle.value == value:
            return

        setting_handle.set_value(value)
        self._rewrite_plugin_config()

        if not setting_handle.runtime and not setting_handle.shader_runtime:
            self._show_restart_hint()
        else:
            self._update_queue.append(
                "setval {}.{} {}".format(self._current_plugin, setting_id, value)
            )

        if setting_handle.type in ("enum", "bool"):
            self._render_current_settings()

    # =========================================================
    # Handlers
    # =========================================================

    def _on_setting_bool_changed(self, setting_id, value):
        self._do_update_setting(setting_id, value == Qt.Checked)

    def _on_setting_enum_changed(self, setting_id, value):
        self._do_update_setting(setting_id, value)

    def _on_setting_power_of_two_changed(self, setting_id, value):
        self._do_update_setting(setting_id, value)

    def _on_setting_slider_changed(self, setting_id, setting_type, bound_objs, value):

        if setting_type == "float":
            real_val = from_slider(value)
        else:
            real_val = int(value)

        self._do_update_setting(setting_id, real_val)

        for obj in bound_objs:
            obj.setValue(real_val)

    def _on_setting_spinbox_changed(self, setting_id, setting_type, bound_objs, value):

        self._do_update_setting(setting_id, value)

        for obj in bound_objs:
            if setting_type == "float":
                obj.setValue(to_slider(value))
            else:
                obj.setValue(int(value))

    # =========================================================

    def _get_widget_for_setting(self, setting_id, setting):

        widget = QWidget()
        layout = QHBoxLayout()
        layout.setAlignment(Qt.AlignCenter)
        widget.setLayout(layout)

        # -----------------------------------------------------

        if setting.type == "bool":

            box = QCheckBox()
            box.setChecked(Qt.Checked if setting.value else Qt.Unchecked)

            qt_connect(
                box,
                "stateChanged(int)",
                partial(self._on_setting_bool_changed, setting_id)
            )

            layout.addWidget(box)

        # -----------------------------------------------------

        elif setting.type in ("float", "int"):

            if setting.type == "float":
                box = QDoubleSpinBox()

                if setting.maxval - setting.minval <= 2.0:
                    box.setDecimals(4)

                box.setSingleStep(abs(setting.maxval - setting.minval) / 100.0)

            else:
                box = QSpinBox()
                box.setSingleStep(max(1, (setting.maxval - setting.minval) // 32))

            box.setMinimum(setting.minval)
            box.setMaximum(setting.maxval)
            box.setValue(setting.value)
            box.setAlignment(Qt.AlignCenter)

            slider = QSlider(Qt.Horizontal)

            if setting.type == "float":
                slider.setMinimum(to_slider(setting.minval))
                slider.setMaximum(to_slider(setting.maxval))
                slider.setValue(to_slider(setting.value))
            else:
                slider.setMinimum(int(setting.minval))
                slider.setMaximum(int(setting.maxval))
                slider.setValue(int(setting.value))

            layout.addWidget(box)
            layout.addWidget(slider)

            qt_connect(
                slider,
                "valueChanged(int)",
                partial(self._on_setting_slider_changed, setting_id, setting.type, [box])
            )

            value_type = "double" if setting.type == "float" else "int"

            qt_connect(
                box,
                "valueChanged(" + value_type + ")",
                partial(self._on_setting_spinbox_changed, setting_id, setting.type, [slider])
            )

        # -----------------------------------------------------

        elif setting.type == "enum":

            box = QComboBox()

            for value in setting.values:
                box.addItem(value)

            box.setCurrentIndex(setting.values.index(setting.value))
            box.setMinimumWidth(145)

            qt_connect(
                box,
                "currentIndexChanged(QString)",
                partial(self._on_setting_enum_changed, setting_id)
            )

            layout.addWidget(box)

        # -----------------------------------------------------

        elif setting.type == "power_of_two":

            box = QComboBox()

            resolutions = [
                str(2 ** i)
                for i in range(1, 32)
                if setting.minval <= 2 ** i <= setting.maxval
            ]

            for value in resolutions:
                box.addItem(value)

            box.setCurrentIndex(resolutions.index(str(setting.value)))
            box.setMinimumWidth(145)

            qt_connect(
                box,
                "currentIndexChanged(QString)",
                partial(self._on_setting_power_of_two_changed, setting_id)
            )

            layout.addWidget(box)

        # -----------------------------------------------------

        else:
            print("ERROR: Unknown setting type:", setting.type)

        return widget

    # =========================================================
    # Unchanged parts (rendering / loading)
    # =========================================================

    def _set_settings_visible(self, flag):
        self.frame_details.setVisible(flag)

    def _render_current_settings(self):
        settings = self._plugin_mgr.settings[self._current_plugin]

        while self.table_plugin_settings.rowCount() > 0:
            self.table_plugin_settings.removeRow(0)

        for name, handle in iteritems(settings):

            if not handle.should_be_visible(settings):
                continue

            row = self.table_plugin_settings.rowCount()
            self.table_plugin_settings.insertRow(row)

            label = QLabel(handle.label)
            self.table_plugin_settings.setCellWidget(row, 0, label)

            item_default = QTableWidgetItem(str(handle.default))
            item_default.setTextAlignment(Qt.AlignCenter)
            self.table_plugin_settings.setItem(row, 1, item_default)

            widget = self._get_widget_for_setting(name, handle)
            self.table_plugin_settings.setCellWidget(row, 2, widget)

    def on_reset_plugin_settings(self):
        """Reset all settings of the current plugin"""

        if not self._current_plugin_instance:
            return

        msg = (
            "Are you sure you want to reset the settings of '"
            + self._current_plugin_instance.name + "'?\n"
            + "This does not reset the Time of Day settings of this plugin.\n\n"
            + "!! This cannot be undone !!"
        )

        reply = QMessageBox.question(
            self, "Warning", msg, QMessageBox.Yes, QMessageBox.No
        )

        if reply == QMessageBox.Yes:

            QMessageBox.information(
                self,
                "Success",
                "Settings have been reset! You may have to restart the pipeline."
            )

            self._plugin_mgr.reset_plugin_settings(self._current_plugin)
            self._rewrite_plugin_config()
            self._show_restart_hint()
            self._load_plugin_list()


    # ---------------------------------------------------------

    def on_plugin_state_changed(self, item):
        """Enable / disable plugin"""

        plugin_id = item._plugin_id
        state = item.checkState() == Qt.Checked

        self._plugin_mgr.set_plugin_enabled(plugin_id, state)

        self._rewrite_plugin_config()
        self._show_restart_hint()


    # ---------------------------------------------------------

    def on_plugin_selected(self):
        """Called when user selects a plugin in list"""

        selected = self.lst_plugins.selectedItems()

        if not selected:
            self._current_plugin = None
            self._current_plugin_instance = None
            self._set_settings_visible(False)
            return

        item = selected[0]

        self._current_plugin = item._plugin_id
        self._current_plugin_instance = self._plugin_mgr.instances[self._current_plugin]

        self._set_settings_visible(True)
        self._render_current_settings()


    def _load_plugin_list(self):

        self._plugin_mgr.unload()
        self._plugin_mgr.load()

        self.lst_plugins.clear()

        for plugin_id, instance in sorted(
                iteritems(self._plugin_mgr.instances),
                key=lambda p: p[1].name):

            item = QListWidgetItem(" " + instance.name)
            item.setCheckState(
                Qt.Checked if self._plugin_mgr.is_plugin_enabled(plugin_id)
                else Qt.Unchecked
            )

            item._plugin_id = plugin_id
            self.lst_plugins.addItem(item)

        self.lst_plugins.setCurrentRow(0)


# =========================================================
# Start app
# =========================================================

app = QApplication(sys.argv)
qt_register_fonts()

configurator = PluginConfigurator()
configurator.show()

app.exec_()
