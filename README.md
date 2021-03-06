<html>
<body>
<h1>CenterNetObjectDetetor</h1>
<font size=3><b>
This is a simple python class CenterNetObjectDetetor based on 
<a href="https://github.com/see--/keras-centernet">see--keras-centernet</a>.<br>
We have added following Python classes to be able to select some specified objects only.<br>
<pre>
<a href="./CenterNetObjectDetector.py">CenterNetObjectDetector</a>
<a href="./DetectedObjectDrawer.py">DetectedObjectDrawer</a>
<a href="./FiltersParser.py">FiltersParser</a>
</pre>

</b></font>

<br>
<h2>1 Installation </h2>
<h3>
1.1 Clone keras-centernet
</h3>
<font size=2>
We have downloaded <a href="https://github.com/see--/keras-centernet">see--keras-centernet</a>,  
which is a repository of keras-implementation of centernet,
and built an inference-environment to detect objects in an image by using the COCO80-pretrained model
 of the keras-centernet.<br>

<br>


<br>
<h3>
1.2 Clone CenterNetObjectDetector
</h3>
Please clone <a href="https://github.com/atlan-antillia/CenterNetObjectDetetor.git">CenterNetObjectDetetor.git</a> in a working folder.
<pre>
>git clone  https://github.com/atlan-antillia/CenterNetObjectDetetor.git
</pre>
Copy the files in that folder to <i>somewhere_cloned_folder/keras-centernet/</i> folder.
<br>

<h3>
1.3 Run CenterNetObjectDetector
</h3>
Please run the following command on your console to detect objects in an image.<br>

<pre>
python CenterNetObjectDetetor.py image_file_or_dir output_dir [filters]
where filters is an optional parameter,and specify a list of classe to be selected something like [car,person]
</pre>

Example
<pre>
>python CenterNetObjectDetetor.py ./images/img.png outputs
</pre>
<img src="./outputs/img.png">
<br>
The image file img.png used here can be downloaded from <a href="https://user-images.githubusercontent.com/11736571/77320690-099af300-6d37-11ea-9d86-24f14dc2d540.png">
here.</a>
<br>
Unfortunately, the detected image will contain a lot of labels (classname) to the detected objects, 
and the original input image wll be covered by them.  <br>
<br>
For the detection results to the same image by EfficientDet, DETR and YOLOv5 are the following.<br>

EfficientDetector:<br>
<img src = "https://github.com/atlan-antillia/EfficientDetector/blob/master/projects/coco/outputs/img.png">
<br>
This EfficientDetector detectioin seems to be better than CenterNet.

DETR:<br>
<img src = "https://github.com/atlan-antillia/DETR/blob/master/detected/img.png">
<br>
This DETR detection shows some NMS failures.
<br> 

YOLOv5:<br>
<img src = "https://github.com/atlan-antillia/Yolov5ObjectDetector/blob/master/output/img.png">
<br>
This YOLOv5 is better and faster than CenterNet.
<br> 
<br>
<h3>
1.4 Some inference examples
</h3>
Example 1<br>
<pre>
>python CenterNetObjectDetetor.py .\images\ShinJuku.png outputs
</pre>

<img src="./outputs/ShinJuku.jpg" >
<br>
Example 2<br>

<pre>
>python CenterNetObjectDetetor.py .\images\ShinJuku2.png outputs
</pre>
<img src="./outputs/ShinJuku2.jpg" >
<br>
Example 3<br>

<pre>
>python CenterNetObjectDetetor.py .\images\Takashimaya2.png outputs
</pre><br>
<img src="./outputs/Takashimaya2.jpg">
<br><br>

<h2>
2 Customizing a visualization process
</h2>
<h3>
2.1 How to save the detected objects information?
</h3>
 We would like to save the detected objects information as a csv file in the following format.<br>
<pre>
id, class,      score, x,   y, width, height
--------------------------------------------
0, car, 0.68, 889, 400, 92, 107
1, car, 0.67, 1319, 617, 308, 279
2, car, 0.65, 1166, 481, 150, 128
3, person, 0.64, 1397, 463, 35, 44
4, car, 0.63, 1027, 381, 84, 88
5, car, 0.61, 1351, 507, 181, 145
6, car, 0.59, 1194, 370, 83, 83
7, person, 0.58, 780, 301, 14, 38
8, car, 0.58, 1552, 651, 336, 263
9, person, 0.58, 752, 458, 34, 79
10, person, 0.58, 1335, 455, 40, 63
11, person, 0.57, 1143, 400, 29, 53
12, car, 0.56, 1242, 903, 327, 167
13, car, 0.55, 1114, 330, 61, 50
14, person, 0.54, 1288, 469, 37, 63
15, car, 0.54, 1015, 307, 57, 54
16, person, 0.54, 634, 566, 44, 53
17, person, 0.53, 639, 465, 31, 73
18, person, 0.53, 867, 375, 21, 38
19, person, 0.52, 871, 497, 40, 43
20, person, 0.52, 1081, 666, 74, 121

</pre>
Furthermore, the number of objects in each class (objects_stats) on the detected objects as csv below.<br>
<pre>
id class     count
-------------------
1,car,17
2,person,28

</pre>
We have implemented CenterNetObjectDetector and DetectedObjectDrawer classes to be able to generate those
csv files.
<br>

<h2>
2.2 How to apply filters to detected objects?
</h2>

Imagine to select some specific objects only by specifying object-classes from the detected objects.<br>
To specify classes to select, we use the list format like this.
<pre>
  [class1, class2,.. classN]
</pre>
For example, you can run the following command to select objects of <i>person</i> and <i>car</i> from <i>images\img.png.</i><br>
<br>
Example 1: filters=[person,car]<br>

<pre>
>python CenterNetObjectDetetor.py images\img.png detected [car,person]
</pre>
In this case, the detected image, objects, objects_stats filenames will become as shown below, with filters [car,person]. 
<pre>
['car','person']img.png
['car','person']img.png_objects.csv
['car','person']img.png_stats.csv
</pre>

You can see the detected image and objects information detected by above command as shown below.<br>
<br>
<br>
['car','person']img.png<br>
<img src="./outputs/['car','person']img.png">
<br><br>

['car','person']img.png_objects.csv<br>
<img src="./outputs/['car','person']img.png_objects.csv.png" width="50%" height="auto">
<br>
<br>
['car','person']img.png_stats.csv<br>

<img src="./outputs/['car','person']img.png_stats.csv.png" width="30%" height="auto">
<br>

</body>

</html>

