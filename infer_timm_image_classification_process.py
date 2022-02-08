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
import os.path

import torch
from ikomia import core, dataprocess
import copy
import timm
from distutils.util import strtobool
from torchvision.transforms import Resize, ToTensor
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from PIL import Image
import urllib
from infer_timm_image_classification.utils import by_batch, polygon2bbox
import numpy as np
from ikomia.core.pycore import CPointF


# Your imports below


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class InferTimmImageClassificationParam(core.CWorkflowTaskParam):

    def __init__(self):
        core.CWorkflowTaskParam.__init__(self)
        # Place default value initialization here
        # Example : self.windowSize = 25
        self.model_name = "resnet18"
        self.pretrained = True
        self.update = False
        self.ckpt = ""
        self.input_size = (224, 224)
        self.class_file = ""

    def setParamMap(self, param_map):
        # Set parameters values from Ikomia application
        # Parameters values are stored as string and accessible like a python dict
        # Example : self.windowSize = int(param_map["windowSize"])
        self.model_name = param_map["model_name"]
        self.pretrained = strtobool(param_map["pretrained"])
        self.ckpt = param_map["ckpt"]
        self.input_size = eval(param_map["input_size"])
        self.class_file = param_map["class_file"]

    def getParamMap(self):
        # Send parameters values to Ikomia application
        # Create the specific dict structure (string container)
        param_map = core.ParamMap()
        # Example : paramMap["windowSize"] = str(self.windowSize)
        param_map["model_name"] = self.model_name
        param_map["pretrained"] = str(self.pretrained)
        param_map["ckpt"] = self.ckpt
        param_map["input_size"] = str(self.input_size)
        param_map["class_file"] = self.class_file
        return param_map


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class InferTimmImageClassification(dataprocess.C2dImageTask):

    def __init__(self, name, param):
        dataprocess.C2dImageTask.__init__(self, name)
        # Add input/output of the process here
        # Example :  self.addInput(dataprocess.CImageIO())
        #           self.addOutput(dataprocess.CImageIO())
        # Add graphics output
        self.addOutput(dataprocess.CGraphicsOutput())
        # Add numeric outputs
        self.addOutput(dataprocess.CBlobMeasureIO())
        self.addOutput(dataprocess.CDataStringIO())


        self.model = None
        # Create parameters class
        if param is None:
            self.setParam(InferTimmImageClassificationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 1

    def predict(self, img):
        srcTensor = self.transform(Image.fromarray(img)).unsqueeze(0)
        out = self.model(srcTensor)
        probabilities = torch.nn.functional.softmax(out[0], dim=0)
        return probabilities

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        # Get parameters :
        param = self.getParam()

        if not self.model or param.update:
            ckpt = None
            if param.pretrained:
                url, filename = (
                    "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt", "imagenet_classes.txt")
                urllib.request.urlretrieve(url, filename)
                with open("imagenet_classes.txt", "r") as f:
                    self.categories = [s.strip() for s in f.readlines()]
            else:
                if os.path.isfile(param.class_file):
                    with open(param.class_file, "r") as f:
                        self.categories = [s.strip() for s in f.readlines()]
                    if os.path.isfile(param.ckpt):
                        ckpt = param.ckpt
                else:
                    print("Impossible to open " + param.class_file)
                    # Step progress bar:
                    self.emitStepProgress()

                    # Call endTaskRun to finalize process
                    self.endTaskRun()

            self.model = timm.create_model(param.model_name, pretrained=param.pretrained, checkpoint_path=ckpt,
                                           num_classes=len(self.categories))
            self.model.eval()
            self.config = resolve_data_config({}, model=self.model)
            self.config["input_size"] = (3, *param.input_size)
            self.transform = create_transform(**self.config)
            param.update = False
        # Get input :
        input = self.getInput(0)
        graphics_input = self.getInput(1)

        # Get output :
        graphics_output = self.getOutput(1)

        # Get image from input/output (numpy array):
        srcImage = input.getImage()
        self.forwardInputImage(0, 0)

        # Init numeric output
        numeric_output1 = self.getOutput(2)
        numeric_output1.clearData()
        numeric_output2 = self.getOutput(3)
        numeric_output2.clearData()

        if srcImage is not None and self.model:
            color = [255, 0, 0]
            # Check if there are boxes as input
            if graphics_input.isDataAvailable():
                polygons = graphics_input.getItems()
                count_item = 0
                # create batch of images containing text
                for item in polygons:
                    crop_img = None
                    if isinstance(item, core.pycore.CGraphicsRectangle):
                        x, y, w, h = item.x, item.y, item.width, item.height
                        pts = [CPointF(x, y),
                               CPointF(x + w, y),
                               CPointF(x + w, y + h),
                               CPointF(x, y + h)]
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        crop_img = srcImage[y:y + h, x:x + w]
                    elif isinstance(item, core.pycore.CGraphicsPolygon):
                        pts = item.points
                        pts_ar = np.array([[pt.x, pt.y] for pt in pts])
                        x, y, w, h = polygon2bbox(pts_ar)
                        crop_img = srcImage[y:y + h, x:x + w]
                    if crop_img is not None:
                        count_item += 1
                        if np.cumprod(np.shape(crop_img)).flatten()[-1] > 0:
                            prob = self.predict(crop_img)
                            class_index = torch.argmax(prob)
                            prop_poly = core.GraphicsPolygonProperty()
                            prop_poly.pen_color = color
                            prop_text = core.GraphicsTextProperty()
                            graphics_box = graphics_output.addPolygon(pts, prop_poly)
                            graphics_box.setCategory(
                                self.categories[class_index])
                            graphics_output.addText(self.categories[class_index], int(x), int(y), prop_text)
                            # Object results
                            results = []
                            confidence_data = dataprocess.CObjectMeasure(
                                dataprocess.CMeasure(core.MeasureId.CUSTOM, "Confidence"),
                                prob[class_index].item(),
                                graphics_box.getId(),
                                self.categories[class_index])
                            box_data = dataprocess.CObjectMeasure(
                                dataprocess.CMeasure(core.MeasureId.BBOX),
                                item.getBoundingRect(),
                                graphics_box.getId(),
                                self.categories[class_index])
                            results.append(confidence_data)
                            results.append(box_data)
                            numeric_output1.addObjectMeasures(results)

            else:
                prob = self.predict(srcImage)
                class_index = torch.argmax(prob)
                msg = self.categories[class_index] + ": {:.3f}".format(prob[class_index])
                graphics_output.addText(msg, 10, 10)
                sorted_data = sorted(zip(prob.flatten().tolist(), self.categories), reverse=True)
                confidences = [str(conf) for conf, _ in sorted_data]
                names = [name for _, name in sorted_data]
                numeric_output2.addValueList(confidences, "probability", names)

        # Call to the process main routine
        # dstImage = ...

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class InferTimmImageClassificationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "infer_timm_image_classification"
        self.info.shortDescription = "Infer timm image classification models"
        self.info.description = "Infer timm image classification models." \
                                "Inference can be done with models pretrained on Imagenet" \
                                "or custom models trained with the plugin " \
                                "train_timm_image_classification."
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python"
        self.info.iconPath = "icons/timm.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "Ross Wightman"
        self.info.article = "PyTorch Image Models"
        self.info.journal = "GitHub repository"
        self.info.year = 2019
        self.info.license = "Apache-2.0 License"
        # URL of documentation
        self.info.documentationLink = "https://rwightman.github.io/pytorch-image-models/"
        # Code source repository
        self.info.repository = "https://github.com/rwightman/pytorch-image-models"
        # Keywords used for search
        self.info.keywords = "timm, infer, image, classification, imagenet, custom"

    def create(self, param=None):
        # Create process object
        return InferTimmImageClassification(self.info.name, param)
