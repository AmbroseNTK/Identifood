<h1>
Identifood - Analyzes your meals and helps you be healthy
</h1>
<h2>
Content
</h2>
<p>

1. [About me](#about)
2. [What is it?](#whatisit)
3. [Why?](#why)
4. [Methodology](#methodology)
5. [Implementation](#implementation)
6. [My result](#myresult)
7. [Plan](#plan)

</p>
<h2>
1. About me <a name="about"></a>
</h2>
<p>
I started coding when I was 13, I learn new technologies by doing some interesting projects and share it to everybody on GitHub. Now, I am Software Engineering senior student of Hoa Sen University, Vietnam. I am also leader of Developers Student Club of Hoa Sen University. I have joined two bootcamp about Machine Learning at Jeju National University, Korea in 2018, 2019 so I have experiences in using some ML framework such as TensorFlow, Keras to make ML Application. I have built some native Android app, they were mini games, questionnaire app.

This project is submitted for **#AndroidDevChallenge** program held by **Google Developers**

</p>

<h2>
2. What is it? <a name="whatisit"></a>
</h2>
<p>

In a nutshell, user will take their meals photos or short video. Then, On-Device ML technology (I would like to use TensorFlow Lite) will analyze these meals and its ingredients to get information about what user have eaten and recording all data so that suggesting what users should eat or not, what exercises they should do.

For example: A user uses this app and takes a picture of his or her breakfast (beefsteak and bread). ML component in the app will answer these questions:
    
    1. What is it? -> beefsteak, bread, salad, tomato (Image Classification)
    2. How much? -> beefsteak: 1 slice, bread: 1, salad: little, tomato: little (Masked Object Detection + Image processing)
    3. Nutrition -> calories, vitamins, carbohydrate,... (nutrition database)

After that, based on user health profile (weight, height, age,...) the app will give sugession like this:
    
    1. Meal's quality index: 90% (meaning: you should have more meals like this and it has very little bad affect to your health)
    2. Workout exercises: push-up: 10 x 3 set OR running: 1000m OR ... (meaning: user can choise one of these exercises and do it to be healthy)

All results will be stored.

</p>

<h2>
3. Why? <a name="why"></a>
</h2>
<p>

This app will track and analyse what people eat in every meals so that suggest them how to be healthy and what kind of food they need every day.

<p>

<h2>
5. Methodology <a name="methodology"></a>
</h2>
<p>
This part will show you important models will be built in ML component for the app. These images and results are referenced from my 

![previous project](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/)

which I have done before

<br>

* **Food Recognition**: It is a model to recognize food in a photo in summary. In my case, it returns result like Bun, Pho, Com, Banh-mi etc. I have downloaded above 600 photo per food on Google Image. After that, I have deleted unneccesary images and just keep correct images. I used these images to train the model.

![FoodRecognitionFlow](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/FoodRecognitionFlow.jpeg)

* **Ingredient Detection**: According to result which I have after applied Food Recognition model I use ingredient Detection model to detect ingredient in food one by one, so that I can calculate its nutrition, predict its taste. Because of kind variation of food, each food has Ingredient Detection model differently. For example, Pho is a popular food in Vietnam, and its ingredient change its nutrition a lot.

![IngredientDetection](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/IngredientDetectionFlow.jpeg)

Pho-bo | Pho-ga
--- | ---
![phobo](http://www.savourydays.com/wp-content/uploads/2013/01/PhoBoHN.jpg) | ![phoga](http://giadinh.mediacdn.vn/zoom/655_361/2014/7-1412602058-tu-lam-pho-ga-7-1412654910607-crop-1412654928337.jpg)

The first photo is Pho-bo means Pho with beef. The second is Pho-ga means Pho with chicken. All of them is Pho, but they have different ingredient. Because of nutrition difference of beef and chicken, so Pho-bo and Pho-ga have different nutrition. So Ingredient Detection is an important step to solve this problem which is common in anothor food's culture.

![FoodRecognition](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/ActivityDiagram.jpg)
</p>

<h2>
6. Implementation <a name="implementation"></a>
</h2>
<p>
In this section, I would like to show you what I have done and how to continue development this project step by step.</br>

<h3>
Step 1. Setup development environment
</h3>
<p>
In this project, you should install some application which is shown in below table.</br>

Tool | Description | Link
--- | --- | ---
![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/python-logo-master-v3-TM.png) | Python is main language to train models | https://www.python.org/downloads/
![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/tensorflow-logo.png) | TensorFlow framework supports all things in ML/DL. If your PC have GPU card, you should install TensorFlow-GPU version to get high performance | https://www.tensorflow.org/install/
![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/Anaconda_Logo.png) | In Windows, some package in unvailable, so you should have Anaconda to install them | https://conda.io/docs/user-guide/install/index.html
</br>
There are three important tools you have to install first, some small tool I will show you after.
</p>

<h3>
Step 2. Prepare data to create Image Recognition model.
</h3>
<p>

There are a lot of ways to collect photos. For me, I refer to collect them on Google Image, because it is the largest search engine, so it contains a lot of photos. The simple way is use a tool that let you download images automatically based on keywords. I have used [this tool](https://github.com/hardikvasa/google-images-download)

. Each food should have different folders. Notice that the folder name is also the label of food, so please check it carefully.</br>

```batch
└───vietnamese_food
    ├───background
    ├───banh bao
    ├───banh mi
    ├───bo bit tet
    ├───bun bo
    ├───com dia
    ├───dau hu
    ├───mi xao
    ├───rau xao
    ├───thit kho tau
    └───trung op la
```
There are 10 common food in Vietnam and backgroud to recognize uneatable things. Here is my sample config</br>

```json
{
    "Records": [
        {
            "keywords": "bun bo",
            "limit":600,
            "type":"photo",
            "format":"jpg",
            "output_directory":"vietnamese_food/",
            "chromedriver":"chromedriver.exe"

        },
        {

        },...
    ]
}
```
After downloading process complete. You need to review all photo, delete error photos or out of topic photo before start training process.
<h3>
Step 3. Prepare dataset for Ingredient Detection model
</h3>
<p>

In this step, you will crop ingredients in food based on photos which you have downloaded in previous step. Before that, you should classify food into groups which have common features. Each group will have a different model. For example, Bun (or noodle rice) in Vietnam has a lot of kinds, so I grouped them into rice-noodle group. This group contains Bun-bo (beef rice noodle), Bun-moc (meatball rice noodle), Bun-ga (chicken rice noodle), etc.. So I would like to put all images of rice-noodle group into rice-noodle folder. After that, I have used [LabelImg](https://github.com/tzutalin/labelImg) to crop ingredients in these images.

![DemoLabelImg](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/LabelImgDemo.PNG)
</br>
After that, dataset folder should have original images and its .xml files which save all information about ingredient boundary rectangle. These .xml files will be converted to csv files. You should take a subset of dataset for test, its about 10% to 20% of dataset. Move these test data into test folder and train data into train folder.
</p>

<h3>
Step 4. Train Food Recognition
</h3>
<p>

Please download [FoodRecognition branch](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/tree/FoodRecognition) in this Git Repository. I have prepared neccessary python script to train Food Recognition model. Download and unzip it, you will have folder structure below

```batch
│
├───ImageRecognizer
│       classify_image.py
│       label_image.py
│       retrain.py
```
Run PowerShell and type this command

```batch
cd <YOUR_IMAGE_RECOGNIZER_DIRECTORY>
```
To start training process

```batch
python retrain.py --image_dir <DIRECTORY_TO_DATASET> --tfhub_module https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/1
```

I haved used mobilenet v2 model, so at tfhub_module I used this link to download model. When training process complete, you will have a model in folder **/tmp/** at root directory. Your model contains two file **"output_graph.pb"** and **"output_labels.txt"**. You need copy them to another place. To use it, run **label_image.py**

```batch
python label_image.py --graph=<DIRECTORY_TO_GRAPH_FILE> --labels=<DIRECTORY_TO_LABELS_FILE> --input_layer=Placeholder --output_layer=final_result --input_height=224 --input_width=224 --image=<YOUR_IMAGE_FILE>
```

</p>

<h3>
Step 5. Train Ingredient Detection
</h3>
<p>

Download [IngredientDetection branch](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/tree/IngredientDetection) to your PC and unzip it. You should focus to folder **models/object_detection**.</br>
Edit file **models/object_detection/training/labelmap.pbtxt**. This file contains all ingredient label so you should edit it to suitable with your case.

```json
item {
  id: 1
  name: '<Label 1>'
}
item {
  id: 2
  name: '<Label 2>'
}
.
.
.
item {
  id: n
  name: '<Label n>'
}
```
Run command to create csv file from dataset.

```batch
# In object_detection folder
python xml_to_csv.py 
```

Edit file **models/generate_tfrecord.py** from line 32. This file help you create tfrecord file which is structured file TensorFlow can understand.

```python
# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Label 1':
        return 1
    elif row_label == 'Label 2':
        return 2
    .
    .
    .
    elif row_label == 'Label n':
        return n
    else:
        return 0
```
Run these command to create tfrecord files.

```batch
python generate_tfrecord.py --csv_input=<TRAIN_FILE_CSV> --image_dir=<TRAIN_FOLDER> --output_path=train.record
python generate_tfrecord.py --csv_input=<TEST_FILE_CSV> --image_dir=<TEST_FOLDER> --output_path=test.record
```

Edit config file of your model which you want to train. For me, I used Inception V2, so I download model from [Here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md), copy unzip folder to models/object_detection, then I edit file models/object_detection/training/faster_rcnn_inception_v2_pets.config at:</br>
* Line 9  : num_classes: <NUMBER_OF_LABEL>
* Line 106: fine_tune_checkpoint: "object_detection/<MODEL_DIRECTORY>/model.ckpt"
* Line 122: input_path: "<TRAIN_RECORD_FILE>"
* Line 124: label_map_path: "<LABELMAP_FILE>"
* Line 136: input_path: "<TEST_RECORD_FILE>"
* Line 138: label_map_path: "<LABELMAP_FILE>"

Before start training, you should compile protobuf file. Follow these commands to do that.

```batch
conda create -n tensorflow1 pip python=3.5
activate tensorflow1
pip install --ignore-installed --upgrade tensorflow-gpu
conda install -c anaconda protobuf
```
In **models/** folder

```batch
protoc --python_out=. .\object_detection\protos\anchor_generator.proto .\object_detection\protos\argmax_matcher.proto .\object_detection\protos\bipartite_matcher.proto .\object_detection\protos\box_coder.proto .\object_detection\protos\box_predictor.proto .\object_detection\protos\eval.proto .\object_detection\protos\faster_rcnn.proto .\object_detection\protos\faster_rcnn_box_coder.proto .\object_detection\protos\grid_anchor_generator.proto .\object_detection\protos\hyperparams.proto .\object_detection\protos\image_resizer.proto .\object_detection\protos\input_reader.proto .\object_detection\protos\losses.proto .\object_detection\protos\matcher.proto .\object_detection\protos\mean_stddev_box_coder.proto .\object_detection\protos\model.proto .\object_detection\protos\optimizer.proto .\object_detection\protos\pipeline.proto .\object_detection\protos\post_processing.proto .\object_detection\protos\preprocessor.proto .\object_detection\protos\region_similarity_calculator.proto .\object_detection\protos\square_box_coder.proto .\object_detection\protos\ssd.proto .\object_detection\protos\ssd_anchor_generator.proto .\object_detection\protos\string_int_label_map.proto .\object_detection\protos\train.proto .\object_detection\protos\keypoint_box_coder.proto .\object_detection\protos\multiscale_anchor_generator.proto .\object_detection\protos\graph_rewriter.proto
```
Then run this command

```batch
protoc object_detection/protos/*.proto --python_out=.
```

We are ready for training. To start, run this command

```batch
python train.py --logtostderr --train_dir=object_detection/training --pipeline_config_path=object_detection/training/<YOUR_MODEL_CONFIG_FILE>
```

Wait for training, In my case, I haved use GPU card, it consumes about 1 second per step. I run above 3000 steps and stop. Here is my result

![IngredientDetectionTrainResult](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/IngredientDetectionTrainingResult.PNG)

To finish, we need extract model from checkpoint file by using export_inference_graph.py

```batch
python export_inference_graph.py --input_type image_tensor --pipeline_config_path training/<YOUR_MODEL_CONFIG_FILE> --trained_checkpoint_prefix training/model.ckpt-<HIGHEST_NUMBER> --output_directory inference_graph
```

Your final model will be saved in folder models/inference_graph</br>
To use the model, edit file Python models/object_detection/Object_detection_image.py at lines

* line 34: IMAGE_NAME = '<INPUT_DIRECTORY>'
* line 50: NUM_CLASSES = <NUMBER_OF_INGREDIENTS>

Save and run it

```batch
python Object_detection_image.py
```

</p>

</p>

<h2>
7. My result <a name="myresult"></a>
</h2>
<p>
Project is developing, I show you current result. It will be updated continuously.</br>

* Food Recognition: I have trained food recogntion model for 10 basic Vietnamese food (Bun, Com, Pho,...). Here is a cross entropy graph.
![CrossEntropy_ImageRecognition](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/CrossEntropyTrainImageDetection.PNG)
Training process have done with result:</br>
![ImageRecognitionTrainResult](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/ImageRecognitionTrainResult.PNG)

Test accuracy is **78.8%**, it is not the best, because I do not have enough dataset. To improve it, I would like to increase number of photo in dataset about 1000 photos per food.</br>
Below is test result:</br>

Photo | Target | Output | Result
--- | --- | --- | ---
![test_bunbo](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/test_bunbo.jpg) | bun bo | bun bo: 0.99073255 | ![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/correct.png)
![test_banhmi](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/test_banhmi.jpg) | banh mi | banh mi 0.99794585 | ![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/correct.png)
![test_com](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/test_com.jpg) | com | com 0.85801125 | ![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/correct.png)
![test_banhbao](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/test_banhbao.jpg) | banh bao | banh bao 0.99786466 | ![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/correct.png)

* Ingredient Detection: Because I do not have enough photo (just 100 photos) so model still has low accuracy.
![test_ingredientDetection1](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/test_object_detection1.PNG)
</br>
Here is graphs

![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/GraphIngredientDetection.PNG)

* Mobile App: It is unfinish. Now, it can capture photo and recognize food, then show food information. Here is some screenshots </br>


![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/Screenshot_20180804-174258.png) | ![](https://github.com/AmbroseNTK/Food-Recognition-For-Blind-People-And-Foreigner/blob/master/img/Screenshot_20180804-174435.png)
--- | ---

</p>

</p>

<h2>
7. Plan <a name="plan"></a>
</h2>
<p>

The section above shown the result I got in previous project. To bring my idea to life as I wrote in the first section and complete **#AndroidDevChallenge**. I need complete the plan below:

    1. Dec 1, 2019 - Jan 1, 2020: Collect more data and retrain ML model with TensorFlow 2.0
    2. Jan 1, 2020 - Feb 1, 2020: Make sure ML models for detect food and recognize food’s ingredients will be done.
    3. Feb 1, 2020 - Mar 1, 2020: Build Android app with full essential functions (capturing foods images, analyzing using TensorFlow Lite, Recording user’s health profile, Suggesting workout exercises)
    4. Mar 1, 2020 - Apr 1, 2020: Testing and improving the performance.
    5. Apr 1, 2020 - May 1, 2020: Finishing and publish the first version to the world. 

</p>
