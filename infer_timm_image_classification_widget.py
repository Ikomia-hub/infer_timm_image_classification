# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.utils import pyqtutils, qtconversion
from infer_timm_image_classification.infer_timm_image_classification_process import InferTimmImageClassificationParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
import timm
from PyQt5.QtCore import Qt
from infer_timm_image_classification.utils import Autocomplete


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class InferTimmImageClassificationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        if param is None:
            self.parameters = InferTimmImageClassificationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        timm_models = timm.list_models(pretrained=True)
        self.combo_model = Autocomplete(timm_models,parent=None, i=True, allow_duplicates=False)
        self.label_model = QLabel("Model name")
        self.gridLayout.addWidget(self.combo_model,0,1)
        self.gridLayout.addWidget(self.label_model,0,0)
        self.combo_model.setCurrentText(self.parameters.model_name)
        self.spin_input_h = pyqtutils.append_spin(self.gridLayout, "Input height", self.parameters.input_size[0],
                                                  min=16)
        self.spin_input_w = pyqtutils.append_spin(self.gridLayout, "Input width", self.parameters.input_size[1],
                                                  min=16)
        self.check_pretrained = pyqtutils.append_check(self.gridLayout, "Pretrained on Imagenet",
                                                       self.parameters.pretrained)
        self.check_pretrained.stateChanged.connect(self.onStateChanged)

        self.browse_ckpt = pyqtutils.append_browse_file(self.gridLayout, label="Checkpoint path",
                                                        path=self.parameters.ckpt)
        self.browse_ckpt.setEnabled(not self.check_pretrained.isChecked())

        self.browse_class_file = pyqtutils.append_browse_file(self.gridLayout, "Class names file",
                                                              self.parameters.class_file)
        self.browse_class_file.setEnabled(not self.check_pretrained.isChecked())

        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.setLayout(layout_ptr)

    def onStateChanged(self, int):
        self.browse_ckpt.setEnabled(not self.check_pretrained.isChecked())
        self.browse_class_file.setEnabled(not self.check_pretrained.isChecked())

    def onApply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.model_name = self.combo_model.currentText()
        self.parameters.pretrained = self.check_pretrained.isChecked()
        self.parameters.update = True
        self.parameters.ckpt = self.browse_ckpt.path
        self.parameters.input_size = (self.spin_input_h.value(), self.spin_input_w.value())
        self.parameters.class_file = self.browse_class_file.path
        # Send signal to launch the process
        self.emitApply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class InferTimmImageClassificationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "infer_timm_image_classification"

    def create(self, param):
        # Create widget object
        return InferTimmImageClassificationWidget(param, None)
