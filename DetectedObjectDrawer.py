#Copyright 2020-2021 antillia.com Toshiyuki Arai
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

#This is based on the COCODrawer
#from keras_centernet.utils.utils import COCODrawer

import math
import cv2
from PIL import ImageFont, ImageDraw, Image
import numpy as np

#2021/01/30

class DetectedObjectDrawer:

  def __init__(self, font_size=24, font="assets/Roboto-Regular.ttf", char_width=14):
 
    self.font_size = font_size
    self.font = ImageFont.truetype(font, font_size)
    self.char_width = char_width

    
  def draw_box_with_class_name(self, img, x1, y1, x2, y2, class_names, cl, id):
    cl = int(cl)
    x1, y1, x2, y2 = int(round(float(x1))), int(round(float(y1))), int(round(float(x2))), int(round(float(y2)))
    h = img.shape[0]
    width = max(1, int(h * 0.002))
    
    name = class_names[int(cl)]
    
    bgr_color = self.get_rgb_color(cl, len(class_names))[::-1]
    # bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), bgr_color, width)
    # font background
    name = str(id) + ":" + name 
    font_width = len(name) * self.char_width
    cv2.rectangle(img, (x1 - math.ceil(width / 2), y1 - self.font_size), (x1 + font_width, y1), bgr_color, -1)

    # text
    pil_img = Image.fromarray(img[..., ::-1])
   
    draw = ImageDraw.Draw(pil_img)
    draw.text((x1 + width, y1 - self.font_size), name, font=self.font, fill=(0, 0, 0, 255))
    img = np.array(pil_img)[..., ::-1].copy()
    return img



  def get_color(self, c, x, max_value, colors=[[1, 0, 1], [0, 0, 1], [0, 1, 1], [0, 1, 0], [1, 1, 0], [1, 0, 0]]):
    # https://github.com/pjreddie/darknet/blob/master/src/image.c
    ratio = (x / max_value) * 5
    i = math.floor(ratio)
    j = math.ceil(ratio)
    ratio -= i
    r = (1. - ratio) * colors[i][c] + ratio * colors[j][c]
    return r


  def get_rgb_color(self, cls, clses):
    offset = cls * 123457 % clses
    red    = self.get_color(2, offset, clses)
    green  = self.get_color(1, offset, clses)
    blue   = self.get_color(0, offset, clses)
    return int(red * 255), int(green * 255), int(blue * 255)


