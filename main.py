# We import the required librairies for the data management system, the analysis module, and the reporting modules

import os
import datetime
import sys

from PIL import Image
import operator
import numpy as np
import cv2

import matplotlib.pyplot as plt #for showing the image
import matplotlib.image as mpimg #for reading the image

import tensorflow as tf
import csv

from collections import defaultdict

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from pathlib import Path

import status

print("Now running with tensorflow v"+str(tf.__version__)+". Please ensure the version is matching with the tree generated with the notebook.")

# Setting up time variables
today = str(datetime.datetime.now().strftime("%d%m%Y_%H%M%S"))
ftoday = str(datetime.datetime.now().strftime("%d/%m/%Y - %H:%M"))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)

# Check if directories are existing
if not os.path.isdir("output"):
    os.mkdir("output")
    print("[INFO] Created output folder.")
if not os.path.isdir("data"):
    os.mkdir("data")
    print("[ERROR] The data folder was non-existant. Created by the application. Please fill (see readme) and retry")
    raise Exception("The data folder was nonexistant.")

class dataManager:
    # This class is in charge of the data management for the application. It holds the current state of the application, and all the 
    # procedures available.
    
    # Initiates the environment with useful variables, and returns it's own state
    def __init__(self, today):
        # The working directory is to be configurated according to the environment.
        # It allows the application to know where it is located right now
        os.chdir(Path(__file__).parent)
        # The status ensures that we don't do operations that might break the DMS
        self.state = status.FS_IDLE
        # The output folder variable keeps the name of the current run output folder. It is set by the elaborateRunArch method.
        self.runOutputFolder = ""
        self.elaborateRunArch(today)
        # Debugging purposes.
        print("[OK] Filesystem started, ready to take over data management. "+str(self))
        print("Current working directory : "+os.getcwd()+"\n")
        
    # Returns a integer representing the current state of the application.
    def getState(self):
        return self.state
    
    # Gives a string with the status of the application. No proper explaination for now.
    def __str__(self):
        return str("Current state of the filesystem : "+str(self.getState()))
    
    # Enter the data folder to read the datasets
    def enterDatasetFolder(self):
        # If we are not in IDLE mode, we force the exitFolder.
        if self.getState() != status.FS_IDLE:
            self.exitFolder()
        #Go to data folder from IDLE
        os.chdir("data")
        self.state = status.FS_RW_DATA
        #debug purposes
        print("Current working directory : "+os.getcwd()+"\n")
        
    # Enter either the Output folder, or the current run output subfolder, depending of the flag status
    def enterOutputFolder(self, globalOut = False):
        # If we are not in IDLE mode, we force the exitFolder.
        if self.getState() != status.FS_IDLE:
            self.exitFolder()
        #Go to output folder or current run subfolder
        if not globalOut:
            os.chdir("output/"+self.runOutputFolder)
            self.state = status.FS_RW_OUTPUT
        else:
            os.chdir("output")
            self.state = status.FS_RW_OUTPUT_GLOBAL
        #debug purposes
        print("Current working directory : "+os.getcwd()+"\n")
      
    # Enter the models folder for the Tensorflow models
    def enterModelsFolder(self):
        # If we are not in IDLE mode, we force the exitFolder.
        if self.getState() != status.FS_IDLE:
            self.exitFolder()
        #Go to models folder
        os.chdir("models")
        self.state = status.FS_RW_MODELS
        #debug purposes
        print("Current working directory : "+os.getcwd()+"\n")
    
    # Create the architecture for the current run subfolder (the folder with the date + the subfolder images) and sets the outFolder flag
    def elaborateRunArch(self, today):
        self.enterOutputFolder(True)
        os.mkdir("run-report-"+today)
        os.mkdir("run-report-"+today+"/images")
        self.runOutputFolder = "run-report-"+today
        self.exitFolder()

    # Exit any folder and go back to IDLE status
    def exitFolder(self):
        #Go to previous folder
        os.chdir("..")
        # If currently in current run subfolder in output, go back once again
        if self.getState() == status.FS_RW_OUTPUT:
            os.chdir("..")
        self.state = status.FS_IDLE
        #debug purposes
        print("Current working directory : "+os.getcwd()+"\n")

    # Scan the data folder and output a data structure representing the data folder, the datasets contained, and all the images included
    def scanFolder(self, html):
        self.enterDatasetFolder()
        # Big loop that lists all the datasets.
        # Each element of the list is a dictionary that contains the name of the subfolder, and a list of all the files in the subfolder
        datasets = os.listdir()
        final = []
        if len(datasets) > 0:
            total = 0
            print("Found following datasets :\n")
            for element in datasets:
                # Listing only folders
                if os.path.isdir(element):
                    batch = {"name":"","img":[]}
                    batch["name"] = element
                    content = os.listdir(element)
                    length = len(content)
                    print("\t"+element+" -> "+str(length)+" elements in this folder")
                    for c in content:
                        batch["img"].append(str(c))
                        print("\t\t"+c)
                    final.append(batch)
                    total += length  
                else:
                    print("\tExcluded "+element+" - Reason : is a file.")
            print("The system found "+str(len(datasets))+" folders and "+str(total)+" elements in total.\n\n") 
            self.exitFolder()
            # print(final)
            html.files(final)
            return final
        else:
            # There was an error (normally only the case if there is no dataset in the folder data)
            print("[ERROR] The system found no datasets in folder ./data - Getting off at next catcher --->口")
            self.exitFolder()
            raise Exception("No dataset is available in data folder.")

class analysisMethods:
    
    # No init for the analysisMethods class - It only contains static methods
    def __init__(self):
        raise Exception("You can't create instances of this class.")
    
    # Holds the main logic for the color analysis - counts the number of pixels for each RGB color (datasets + global)
    def imageColorAnalysis(fs, datasets, html):
        # Enter data folder
        fs.enterDatasetFolder()
        
        # Setup the global count
        count = {}
        
        # Iterate among all datasets
        for d in range(0, len(datasets)):
            # For this particular dataset, get his name, navigate and look at the images
            ds = datasets[d]
            os.chdir(ds["name"])
            
            # Setup the count for each dataset
            countDs = {}
            
            # For each image in the dataset, do
            for e in range(0,len(ds["img"])):
                # print(ds["img"][e])
                # Open the image
                im = Image.open(ds["img"][e])
                # Setup the count for this particular image
                countImg = {}
                # For each pixel found
                for i in im.getdata():
                    # If the color didn't exist in one or multiple lists, we add it
                    if i not in count:
                        count[i] = 0
                    if i not in countDs:
                        countDs[i] = 0
                    if i not in countImg:
                        countImg[i] = 0
                    # We increment each counter
                    count[i] += 1
                    countDs[i] += 1
                    countImg[i] += 1
                # Close the image
                im.close()
                
            # All images of the dataset are treated, we sort the dataset array
            sorted_count = sorted(countDs.items(), key=operator.itemgetter(1), reverse=True)
            
            # We come back to the data folder, and force the system to switch to the current run output folder
            os.chdir("..")
            fs.enterOutputFolder()
            
            # We write a csv file with all the informations, and we write in the html file
            csvReporter.new(ds["name"]+"-color", sorted_count, 30) 
            html.wCol(ds["name"], sorted_count, 30)
            
            # We come back to the data folder
            fs.enterDatasetFolder()
            
        # The analysis is finished, we now have the global result. We sort it
        sorted_count_gen = sorted(count.items(), key=operator.itemgetter(1), reverse=True)

        # Same as for the datasets for the csv and the html file. We go to our CR output, we write everything, and we leave
        fs.enterOutputFolder()
        csvReporter.new("global-color", sorted_count_gen, 30)
        html.wCol("global", sorted_count_gen, 30)
        fs.exitFolder()
        
    # This is where the magic happens...
    # All the functions below are based on the Tony607's Colab notebook - https://github.com/Tony607/object_detection_demo
    # I don't have the required knowledge to create my own Object Detection model and my own algorithm
    # However, the functions have been adaptated in order to have a painless integration with my application 
    # (Modification of the data formatting, modification of several variables...)
    
    # Special method for categories
    def get_num_classes(pbtxt_fname):
        from object_detection.utils import label_map_util
        label_map = label_map_util.load_labelmap(pbtxt_fname)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=90, use_display_name=True)
        category_index = label_map_util.create_category_index(categories)
        return len(category_index.keys())

    # Convert an actual image into a numpy array,in order to apply the tensorflow model
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

    # Detect the objects and outputs various informations for one image. Used in applyAnalysis
    def run_inference_for_single_image(image, graph):
        with graph.as_default():
            with tf.Session() as sess:
                # Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {
                    output.name for op in ops for output in op.outputs}
                tensor_dict = {}
                for key in [
                    'num_detections', 'detection_boxes', 'detection_scores',
                    'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                            tensor_name)
                if 'detection_masks' in tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(
                        tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(
                        tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(
                        tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [
                                               real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [
                                               real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                        detection_masks, detection_boxes, image.shape[0], image.shape[1])
                    detection_masks_reframed = tf.cast(
                        tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    tensor_dict['detection_masks'] = tf.expand_dims(
                        detection_masks_reframed, 0)
                image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

                # Run inference
                output_dict = sess.run(tensor_dict,
                                       feed_dict={image_tensor: np.expand_dims(image, 0)})

                # all outputs are float32 numpy arrays, so convert types as appropriate
                output_dict['num_detections'] = int(
                    output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                    'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict

    # Object analysis launcher method. It takes the filesystem, the list of images and the html file
    # And outputs all the images with the detected images for each model in the images folder in current output subfolder
    # Also edits the html file to add the files
    def applyAnalysis(fs, datasets, html):
        
        # For the future, reorganize this part of the code
        # Get all the models available
        fs.enterModelsFolder()  
        models = os.listdir()
        
        print(models)
        
        fs.exitFolder()
        
        for element in models:
            # Apply the current object recognition model selected
            print("Now looking for : "+element)

            # Verify that the files that we need to apply the model exists
            pb_fname = "./models/"+element+"/frozen_inference_graph.pb"
            assert os.path.isfile(pb_fname), '[ERR] `{}` does not exist'.format(pb_fname)

            label_map_pbtxt_fname = "./models/"+element+"/label_map.pbtxt"
            assert os.path.isfile(pb_fname), '[ERR] `{}` does not exist'.format(pb_fname)

            # Path to frozen detection graph. This is the actual model that is used for the object detection.
            PATH_TO_CKPT = pb_fname

            # List of the strings that is used to add correct label for each box.
            PATH_TO_LABELS = label_map_pbtxt_fname
                              
                
            # Get the different classes
            num_classes = analysisMethods.get_num_classes(label_map_pbtxt_fname)

            # Loads the detection graph in memory with the accurate file
            detection_graph = tf.Graph()
            with detection_graph.as_default():
                od_graph_def = tf.GraphDef()
                with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

            # Configuration variables for the graph
            label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
            categories = label_map_util.convert_label_map_to_categories(
                label_map, max_num_classes=num_classes, use_display_name=True)
            category_index = label_map_util.create_category_index(categories)

            # Enters the dataset folder
            fs.enterDatasetFolder()

            # Sets the imgCount for the html display
            imgCount = 1

            # For each subfolder in our data folder
            for d in range(0, len(datasets)):
                # Enter the dataset
                ds = datasets[d]
                os.chdir(ds["name"])

                # For each image
                for e in range(0,len(ds["img"])):
                    # Open the image
                    image = Image.open(ds["img"][e])
                    (width, height) = image.size

                    # the array based representation of the image will be used later in order to prepare the
                    # result image with boxes and labels on it.
                    image_np = analysisMethods.load_image_into_numpy_array(image)

                    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                    image_np_expanded = np.expand_dims(image_np, axis=0)

                    # Actual detection.
                    output_dict = analysisMethods.run_inference_for_single_image(image_np, detection_graph)

                    # Visualization of the results of a detection.
                    vis_util.visualize_boxes_and_labels_on_image_array(
                        image_np,
                       output_dict['detection_boxes'],
                       output_dict['detection_classes'],
                       output_dict['detection_scores'],
                       category_index,
                       instance_masks=output_dict.get('detection_masks'),
                       use_normalized_coordinates=True,
                        line_thickness=8)

                    # Displays the final image with the box visualisation
                    plt.figure(figsize=IMAGE_SIZE)
                    plt.imshow(mpimg.imread(ds["img"][e]))
                    plt.imshow(image_np)
                    plt.show()

                    # We convert this visualisation to an actual image
                    fimage = Image.fromarray(image_np)

                    # Get out and save in output folder
                    print("Done analysis for image "+ds["img"][e])
                    os.chdir("..")
                    fs.enterOutputFolder()
                    fimage.save("images/"+str(element)+"-"+str(imgCount)+".jpeg")

                    # Get back in for the next iteration
                    fs.enterDatasetFolder()
                    os.chdir(ds["name"])
                    
                    # Next image
                    imgCount += 1
                    #print(image_np, output_dict['detection_classes'], output_dict['detection_scores'])

                # We are out of here, next dataset
                os.chdir("..")

            # We leave the data folder and write the code to see the images on our html file
            fs.exitFolder()
            html.pPic(imgCount, element)

# HTML Snippet - Beginning of the file
html_head = """
        <html lang='en'>
            <head>
                <meta charset='utf-8'>
                <meta name='viewport' content='width=device-width, initial-scale=1, shrink-to-fit=no'>
                <meta name='author' content='HTML report generated by AutoBot (provisional)'>
                <meta name='description' content='Report generated on """+ftoday+""" by AutoBot. It contains color informations and recognized objects for provided batchs.'>
                <title>Execution report - """+ftoday+"""</title>
                <link rel='stylesheet' href='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/css/bootstrap.min.css' integrity='sha384-ggOyR0iXCbMQv3Xipma34MD+dH/1fQ784/j6cY/iJTQUOhcWr7x9JvoRxT2MZw1T' crossorigin='anonymous'>
            </head>
            <body>
                <div class='container'>
                    <h1>Drawing analysis - Execution report</h1>
                    <p>Report generated on """+ftoday+"""</p><hr />"""

# HTML Snippet - Ending of the file
html_footer = """   
                </div>
                <script src='https://code.jquery.com/jquery-3.3.1.slim.min.js' integrity='sha384-q8i/X+965DzO0rT7abK41JStQIAqVgRVzpbzo5smXKp4YfRvH+8abtTE1Pi6jizo' crossorigin='anonymous'></script>
                <script src='https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.14.7/umd/popper.min.js' integrity='sha384-UO2eT0CpHqdSJQ6hJty5KVphtPhzWj9WO1clHTMGa3JDZwrnQq4sF86dIHNDz0W1' crossorigin='anonymous'></script>
                <script src='https://stackpath.bootstrapcdn.com/bootstrap/4.3.1/js/bootstrap.min.js' integrity='sha384-JjSmVgyd0p3pXB1rRibZUAYoIIy6OrQ6VrjIEaFf/nJGzIxFDsf4x0xIM+B07jRM' crossorigin='anonymous'></script>
            </body>
        </html> """

# Handles the writing of the HTML file. Since the buffer is too small, we regularly flush it to write more data.
class htmlReporter:
    
    # Opens the file with the snippet (needs to correctly setup the filesystem before, see main)
    def __init__(self):
        self.f = open("generated-report.html","w")
        self.f.write(html_head)
        self.f.flush()
        print("[OK] Opened a new HTML file in writing mode")
    
    # Close the file with the snippet
    def close(self):
        self.f.write(html_footer)
        self.f.close()
        print("[OK] Closed HTML file output\\run-report-"+today+"\\generated-report.html with success. You can now load the file in your web browser.")
    
    # Handles the error reporting in the file
    def error(self, e):
        self.f.write("""<div class='alert alert-danger' role='alert'>
          A critical error has happened. The process halted. Reason : """+str(e)+"""
        </div>""")
        
    # Handles the file listing
    def files(self, fList):
        self.f.write("<h2>Files in the data folder</h2><div style='justify-content: center;width: 100%;'>\n")
        for c in range(0,len(fList)):
            # We list all the files for each subfolder
            self.f.write("<h3>Dataset "+fList[c]["name"]+"</h3>\n<ul>\n")
            for i in range(0, len(fList[c]["img"])):
                # Writing all the file names in a ul
                self.f.write("<li>"+fList[c]["img"][i]+"</li>\n")
            self.f.write("</ul>\n")
        self.f.write("</div>\n<br />\n\n")
        self.f.flush()

    # Handles the color listing
    def wCol(self, bName, cList, maxNb):
        self.f.write("<h2>Color frequency for dataset : "+bName+"</h2><div style='display: flex;justify-content: center;width: 100%;'>\n")
        for c in range(0,maxNb):
            # Convert the rgb to actual hexadecimal color code
            h = "#%02x%02x%02x" % cList[c][0]
            self.f.write("<div style='width: 100; height: 100; background-color:{}'>{}</div>\n".format(h, cList[c][1]))
            # Handles the auto formatting (no more than 7 colors per line)
            if (c+1) % 7 == 0:
                self.f.write('''</div>\n<br />\n<div style='display: flex;justify-content: center;width: 100%;'>\n''')
            self.f.flush()
        self.f.write("</div>\n<br />\n<p>CSV file -----> <a href='"+bName+"-color.csv'>"+bName+"-color.csv</a></p>\n")
        self.f.flush()
    
    # Handles the image listing
    def pPic(self, pictures, name):
        self.f.write("<h2>Results for object recognition (looking for : "+name+")</h2><div style='display: flex;justify-content: center;width: 100%;'>")
        for c in range(1,pictures-1):
            self.f.write('''<div class="card" style="width: 18rem;">\n''')
            self.f.write("<a href='images/"+str(name)+"-"+str(c)+".jpeg'><img src='images/"+str(name)+"-"+str(c)+".jpeg' class='card-img-top' alt='"+str(name)+"-"+str(c)+"' \></a>\n")   
            self.f.write('''</div>\n''')
            # Handles the auto formatting (no more than 5 images per line)
            if c % 5 == 0:
                self.f.write('''</div><br /><div style='display: flex;justify-content: center;width: 100%;'>''')
            self.f.flush()
        self.f.write("</div><br />")
        self.f.flush()

# This class handles the creation of csv files for data exportation
class csvReporter:
    def new(name, cList, maxNb, fieldnames = ['rgb_tuple', 'number_of_pixels']):
        with open(name+'.csv', 'w', newline='') as csvfile:
            f = csv.DictWriter(csvfile, fieldnames=fieldnames)
            f.writeheader()
            if maxNb <= len(cList):
                for i in range(0, maxNb):
                    f.writerow({'rgb_tuple': cList[i][0], 'number_of_pixels': cList[i][1]}) 

try:  
    # Setting up Filesystem
    fs = dataManager(today)
    
    # Configuring the html reporter module
    fs.enterOutputFolder()
    html = htmlReporter()
    fs.exitFolder()
    
    # Get the listing for the data
    datasets = fs.scanFolder(html)
    
    # Apply the analysis (here color and object recognition, can add more in the future)
    analysisMethods.imageColorAnalysis(fs, datasets, html)
    analysisMethods.applyAnalysis(fs, datasets, html)

except Exception as e:
    print("[HALT] An error occured during the execution of the process: ",e)
    html.error(e)
finally:
    html.close()