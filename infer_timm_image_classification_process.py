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
import copy
import os.path
import urllib

import torch

from PIL import Image

import numpy as np

from ikomia import utils, core, dataprocess

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferTimmImageClassificationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        self.model_name = "resnet18"
        self.use_pretrained = True
        self.update = False
        self.model_weight_file = ""
        self.input_size = (224, 224)
        self.class_file = ""

    def set_values(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        self.model_name = param_map["model_name"]
        self.use_pretrained = utils.strtobool(param_map["use_pretrained"])
        self.model_weight_file = param_map["model_weight_file"]
        self.input_size = eval(param_map["input_size"])
        self.class_file = param_map["class_file"]

    def get_values(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = {
            "model_name": self.model_name,
            "use_pretrained": str(self.use_pretrained),
            "model_weight_file": self.model_weight_file,
            "input_size": str(self.input_size),
            "class_file": self.class_file
        }
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferTimmImageClassification(dataprocess.CClassificationTask):

    def __init__(self, name, param):
        dataprocess.CClassificationTask.__init__(self, name)

        self.model = None
        self.categories = None
        self.config = None

        # Create parameters class
        if param is None:
            self.set_param_object(InferTimmImageClassificationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

        self.model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "weights")


    @staticmethod
    def polygon2bbox(pts):
        x = np.min(pts[:, 0])
        y = np.min(pts[:, 1])
        w = np.max(pts[:, 0]) - x
        h = np.max(pts[:, 1]) - y
        return [int(x), int(y), int(w), int(h)]

    def get_progress_steps(self):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def predict(self, img):
        src_tensor = self.transform(Image.fromarray(img)).unsqueeze(0)
        out = self.model(src_tensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        return probabilities

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        # Get parameters :
        param = self.get_param_object()

        if not self.model or param.update:
            ckpt = None
            if param.model_weight_file == "":
                # ImageNet pre-trained model
                param.use_pretrained = True

            if param.use_pretrained:
                class_filename = os.path.join(os.path.dirname(__file__), "imagenet_classes.txt")
                if not os.path.isfile(class_filename):
                    url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
                    urllib.request.urlretrieve(url, class_filename)

                with open(class_filename, "r") as f:
                    self.categories = [s.strip() for s in f.readlines()]
                    self.set_names(self.categories)
            else:
                if os.path.isfile(param.class_file):
                    with open(param.class_file, "r") as f:
                        self.categories = [s.strip() for s in f.readlines()]

                    if os.path.isfile(param.model_weight_file):
                        ckpt = param.model_weight_file
                else:
                    print("Unable to open " + param.class_file)
                    # Step progress bar:
                    self.emit_step_progress()
                    # Call end_task_run to finalize process
                    self.end_task_run()

            torch.hub.set_dir(self.model_folder)
            self.model = timm.create_model(param.model_name,
                                           pretrained=param.use_pretrained,
                                           checkpoint_path=ckpt,
                                           num_classes=len(self.categories))
            self.model.eval()
            self.config = resolve_data_config({}, model=self.model)
            self.config["input_size"] = (3, *param.input_size)
            self.transform = create_transform(**self.config)
            param.update = False

        # Get input :
        img_input = self.get_input(0)

        # Get image from input/output (numpy array):
        src_image = img_input.get_image()
        self.forward_input_image(0, 0)

        # Check if there are boxes as input
        if not self.is_whole_image_classification():
            input_objects = self.get_input_objects()
            for obj in input_objects:
                roi_img = self.get_object_sub_image(obj)
                if roi_img is None:
                    continue

                prob = self.predict(roi_img)
                class_index = torch.argmax(prob).item()
                self.add_object(obj, class_index, prob[class_index].item())
        else:
            prob = self.predict(src_image)
            sorted_data = sorted(zip(prob.flatten().tolist(), self.categories), reverse=True)
            confidences = [str(conf) for conf, _ in sorted_data]
            names = [name for _, name in sorted_data]
            self.set_whole_image_results(names, confidences)

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferTimmImageClassificationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_timm_image_classification"
        self.info.short_description = "Infer timm image classification models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Classification"
        self.info.icon_path = "icons/timm.png"
        self.info.version = "1.2.0"
        self.info.authors = "Ross Wightman"
        self.info.article = "PyTorch Image Models"
        self.info.journal = "GitHub repository"
        self.info.year = 2019
        self.info.license = "Apache-2.0 License"
        # Ikomia API compatibility
        self.info.min_ikomia_version = "0.11.1"
        # Python compatibility
        self.info.min_python_version = "3.8.0"
        # URL of documentation
        self.info.documentation_link = "https://rwightman.github.io/pytorch-image-models/"
        # Code source repository
        self.info.repository = "https://github.com/Ikomia-hub/infer_timm_image_classification"
        self.info.original_repository = "https://github.com/rwightman/pytorch-image-models"
        # Keywords used for search
        self.info.keywords = "timm, infer, image, classification, imagenet, custom"
        self.info.algo_type = core.AlgoType.INFER
        self.info.algo_tasks = "CLASSIFICATION"

    def create(self, param=None):
        # Create process object
        return InferTimmImageClassification(self.info.name, param)
