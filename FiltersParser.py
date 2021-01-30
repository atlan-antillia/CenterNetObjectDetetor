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

import os

class FiltersParser:

  # Specify a str_filters string like this "[person,motorcycle]" ,
  # If needed, please specify your own classes.
  # default classes = COCO_CLASSES
  def __init__(self, classes):
      
      self.str_filters  = None
      self.classes      = classes
      self.filters  = []


  def parse(self, filters):
    if filters is None or filters == "":
      return None
      
    self.str_filters = filters
    self.filters = []
    if self.str_filters != None:
        tmp = self.str_filters.strip('[]').split(',')
        if len(tmp) > 0:
            for e in tmp:
                e = e.lstrip()
                e = e.rstrip()
                if e in self.classes :
                  self.filters.append(e)
                else:
                  raise Exception("Invalid label(class)name {}".format(e))
                  
    return self.filters


  def get_ouput_filename(self, input_image_filename, image_out_dir):
        rpos  = input_image_filename.rfind("/")
        fname = input_image_filename
        if rpos >0:
            fname = input_image_filename[rpos+1:]
        else:
            rpos = input_image_filename.rfind("\\")
            if rpos >0:
                fname = input_image_filename[rpos+1:]
          
        
        abs_out  = os.path.abspath(image_out_dir)
        if not os.path.exists(abs_out):
            os.makedirs(abs_out)

        filname = ""
        if self.str_filters is not None:
            filname = self.str_filters.strip("[]").replace("'", "").replace(", ", "_")
            if len(filname) != 0:
              filname += "_"
        
        output_image_filename = os.path.join(abs_out, filname + fname)
        return output_image_filename

