<img width="1494" alt="image" src="https://github.com/angryraptor108/HC_FaceRecognition/blob/main/facerecognizeranimation.gif">

# Haar Cascades Face Recognizer

Use this to train your own haar cascades model to recognize familiar faces with training data (test images). 

### Quickstart
```sh
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Upload images under **images** folder, with each person under a different subfolder (titled with the heading required during detection).

### Training
```sh
python faces-train.py
```

In the terminal, you should see images for which a face was not recognized. Delete these under **images**. The **testimages** folder provides the detected face area. If this area isn't the actual face, delete these images as well. Run the training as many times as desired. 

### Running the Model
```sh
python faces.py
```

By default, all non-recognized people are called **random lurkers**. This can be changed in line 52 in **faces.py**. 
