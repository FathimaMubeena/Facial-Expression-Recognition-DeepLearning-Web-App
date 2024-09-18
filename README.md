# Predicting Facial-Expression-Recognition using Convolutional Neural Network (CNN)
## Project Purpose:
Facial expression recognition is a key component in understanding human emotions and interactions, with applications ranging from human-computer interaction to mental health assessment. This project aims to develop a convolutional neural network (CNN) that can accurately classify facial expressions into one of seven emotion categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral..

**Dataset:**
The data consists of 48x48 pixel grayscale images of faces. 
The faces have been automatically registered so that the face is more or less centered and occupies about the same amount of space in each image. 
The task is to categorize each face based on the emotion shown in the facial expression in to one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).

### Business Case
The create a ML model that acurately predict a the emotions in the images of type JPEG, JPE.
This project underscores the power of machine learning in understanding complex emotional cues from visual data, with the potential to enhance both human-computer interactions and emotional intelligence in AI systems.

### Goal:
The primary goal of this project is to build an emotional classifier that can effectively identify human emotions based on facial expressions. Using deep learning techniques with Keras, the model is trained to maximize classification accuracy across these seven emotion categories. The final deliverable includes a fully trained CNN model capable of real-time emotion detection, which can be integrated into various applications such as customer sentiment analysis, security systems, or assistive technologies.

### Deliverables:
1. A Keras neural network model that predicts the emotions of people in the picture.
2. A web application that allows to predict the emotions of the people in the Image and predict the emotional percentage.

# How to Run the Project
1. Clone the repository:
   ```shell
   git clone
   ```
2. Create a virtual environment:
   ```shell
    python3 -m venv facial_expression_detection_env
    ```
2. Activate the virtual environment:  
   On macOS:
      ```shell
      source facial_expression_detection_env/bin/activate
      ```
3. Install the required packages:
   ```shell
   pip install -r requirements.txt
   ```
4. Run the Streamlit app:
    ```shell
    streamlit run src/main/deploy/deploy.py
    ```
5. Open the link in your browser to view the app.
6. Deactivate the virtual environment:
   ```shell
   deactivate
   ```

## Deploy the Model on a Cloud Platform (Streamlit Cloud, Heroku, AWS):
Once the app is running locally, you may want to deploy it on a cloud service like Streamlit Cloud, Heroku, or AWS to make it accessible via the web.

### For Streamlit Cloud:
1. Sign up for a free account at Streamlit Cloud.
2. Push your code and model to a GitHub repository.
3. Create a new app on Streamlit Cloud and connect it to your repository.
4. Deploy the app directly from GitHub.

```shell
streamlit run src/main/deploy/deploy.py
```

## Results and Model Evaluation:

Accuracy:
![img.png](src%2Fmain%2Fresources%2Fimg.png)

### Next Steps:
1. Data Augmentation: Increase the size of the training set by applying transformations to the images like rotation, scaling, and flipping.
2. Use Transfer Learning: Use a pre-trained CNN model like `VGG16`, `ResNet50`, or `MobileNetV2` and fine-tune the last few layers.
2. Regularization: Add dropout layers to prevent overfitting and improve the model's generalization.
3. Hyperparameter Tuning: Use GridSearchCV to find the best hyperparameters for the model.
4. Model Interpretability: Use techniques like SHAP values to explain the model's predictions to end users.
5. Deploy the model to a cloud platform like AWS, Azure, or Google Cloud Platform.
6. Monitor the model's performance and retrain it periodically to keep it up-to-date.