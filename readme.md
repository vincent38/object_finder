#  :mag:オブジェクトを見つけましょう
*Let's find objects !*

## What is this software ?
This is a small project written originally for a non-trivial data analysis seminar taken back when I was studying at Kyoto University in spring-summer 2019.

The goal was to generate a report on the most frequent colors in several batchs of pictures, and to detect 
specific objects present on those images. 

## How to run the script ?
In the folder data, create at least one folder (i. e. batch1, no space), and put the images that you want to analyze. You can create as much batches as you want, with any classification order.

Then, in the models folder, put the models of objects that you want to recognize. In the provided example, there is a folder named person that contains a pb and a pbtxt file. This model recognizes faces of persons in drawings, though I have not used enough data to improve the model.

You can put your own model by following the same architecture (create a folder with the name of your choice, then put the frozen_interface_graph.pb and the label_map.pbtxt of your model that you will have previously generated - see the object_detection_demo fork for a small example).

Then, launch the main.py, and let the program generate the report. You will find the generated HTML reports, the CSV files and the images with the objects found on the output folder, organized by date. You will need an internet connection to display the HTML report correctly.

## Technical details

* Model used to generate the modules : ssd_mobilenet_v2_coco_2018_03_29
* Please ensure that you are using the same tensorflow version for app usage and module generation !
* Please ensure that the folder data is present and contains at least one fubfolder with at least one image.

## Warning
This program is provided at is it with no warranty. You can edit it to fit your needs, but please tell me beforehand !
Please do not use it for commercial purposes ;) ;)