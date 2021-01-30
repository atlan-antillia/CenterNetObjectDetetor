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

# This is based on keras_centernet/bin/ctdet_image.py
import os
import sys
import traceback

#!/usr/bin/env python3
import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm
from glob import glob

from keras_centernet.models.networks.hourglass import HourglassNetwork, normalize_image
from keras_centernet.models.decode import CtDetDecode


from keras_centernet.utils.letterbox import LetterboxTransformer

from DetectedObjectDrawer import DetectedObjectDrawer

from FiltersParser import FiltersParser

class CenterNetObjectDetector:

  def __init__(self, classes_path='./dataset/coco/classes.txt', weights='ctdet_coco'):
    self.resolution = (512,512)
    
    self.class_names = self.read_classes(classes_path)
    
    self.weigts      = weights
    kwargs = {
      'num_stacks': 2,
      'cnv_dim': 256,
      'weights': weights,   #'ctdet_coco',
      'inres'  : self.resolution,
    }
    heads = {
      'hm': 80,  # 3
      'reg': 2,  # 4
      'wh':  2   # 5
    }
    self.model = HourglassNetwork(heads=heads, **kwargs)
    self.model = CtDetDecode(self.model)
    

  def get_classes(self):
    return self.class_names
    
    
  def read_classes(self, classes_path):
    classes = None
    if not os.path.exists(classes_path):
      raise Exception("Not found {}".format(classes_path))
    with open(classes_path) as f:
      classes = [s.strip() for s in f.readlines()]
          
    print(classes)
    return classes
    

  def detect_all(self, image_dir, output, filters=None):
    image_files = []

    if os.path.isdir(image_dir):
    
      image_files.extend(glob(os.path.join(image_dir, "*.png")) )
      image_files.extend(glob(os.path.join(image_dir, "*.jpg")) )

      #print("image_files {}".format(image_files) )
          
      for filename in image_files:
        self.detect(filename, output, filters)
 
      
  def detect(self, filename, output, filters=None):
    print("=== detect {} {} {}".format(filename, output, filters))
    self.output_dir    = output
    self.MIN_THRESHOLD = 0.4
    self.NL  = "\n"
    self.SEP = ","
    parser = FiltersParser(self.class_names)
    filters = parser.parse(filters)
    
    drawer = DetectedObjectDrawer()

    img = cv2.imread(filename)
    letterbox_transformer = LetterboxTransformer(self.resolution[0], self.resolution[1])
    pimg = letterbox_transformer(img)
    pimg = normalize_image(pimg)
    pimg = np.expand_dims(pimg, 0)
    detections = self.model.predict(pimg)[0]
          
    objects_detail = []
    objects_stats  = {}
    
    id = 0
    
    for d in detections:
      x1, y1, x2, y2, score, cl = d
      if score < self.MIN_THRESHOLD:
        break

      name = self.class_names[int(cl)]

      if (filters is None) or (filters is not None and name in filters):
        x1, y1, x2, y2 = letterbox_transformer.correct_box(x1, y1, x2, y2)
        img = drawer.draw_box_with_class_name(img, x1, y1, x2, y2, self.class_names, cl, id)

        s = format(score, '.2f')
        
        detected_object = (id, name, s, int(x1), int(y1), int(x2-x1), int(y2-y1) )
        id +=  1
        
        #Update the objects_detail list
        objects_detail.append(detected_object)

        #Update the objects_stats
        if name not in objects_stats:
          objects_stats[name] = 1
        else:
          count = int(objects_stats[name]) 
          objects_stats.update({name: count+1})
      
    sfilters = self.filters_to_string(filters)
    out_filename = self.output_dir + "/" + sfilters + os.path.basename(filename)
        
    #out_filename = os.path.join(self.output_dir, os.path.basename(filename))
    cv2.imwrite(out_filename, img)
    print("=== Saved image file {}".format(out_filename))
    
    self.save_detected_objects(out_filename, objects_detail)
    self.save_objects_stats(out_filename,    objects_stats)
    print("=== Detected objects count {}".format(id))
    return id


  def filters_to_string(self, filters):
    sfilters = ""
    if filters == None:
      sfilters = ""
    else:
      sfilters = str(filters).replace(" ", "")
    return sfilters
          
      
  def save_detected_objects(self, image_name, detected_objects):

    detected_objects_csv = image_name + "_objects.csv"
    print("=== {}".format(detected_objects_csv))
    
    with open(detected_objects_csv, mode='w') as f:
      header = "id, class, score, x, y, w, h" + self.NL
      f.write(header)

      for item in detected_objects:
        line = str(item).strip("()").replace("'", "") + self.NL
        #print(line)
        f.write(line)
   
    print("=== Saved detected_objects {}".format(detected_objects_csv))


  def save_objects_stats(self, image_name, objects_stats):
    objects_stats_csv = image_name + "_stats.csv"
    print("=== {}".format(objects_stats_csv))
    
    with open(objects_stats_csv, mode='w') as s:
       header = "id, class, count" + self.NL
       s.write(header)
       
       for (k,v) in enumerate(objects_stats.items()):
         (name, value) = v
         line = str(k +1) + self.SEP + str(name) + self.SEP + str(value) + self.NL
         s.write(line)
    print("=== Saved objects_stats {}".format(objects_stats_csv))
  


if __name__ == '__main__':

  try:
     if len(sys.argv) < 3:
        raise Exception("Usage: {} image_file_or_dir output_dir filters".format(sys.argv[0]))
        
     image_file = None
     image_dir  = None
     output_dir = None
     filters = None  # classnames_list something like this "[person,car]"
     
     if len(sys.argv) >= 2:
       input = sys.argv[1]
       if not os.path.exists(input):
         raise Exception("Not found input {}".format(input))
       if os.path.isfile(input):
         image_file = input
       else:
         image_dir  = input

     if len(sys.argv) >= 3:
       output_dir = sys.argv[2]
       if not os.path.exists(output_dir):
         os.makedirs(output_dir)
  
     if len(sys.argv) == 4:
       filters = sys.argv[3]

     detector = CenterNetObjectDetector()
 
     if image_dir is not None:
        detector.detect_all(image_dir, output_dir, filters)
       
     if image_file is not None:
        detector.detect(image_file, output_dir, filters)

  except:
    traceback.print_exc()
    
