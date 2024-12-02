# Deepfake Video Detection Using 3D CNN
**Overview**

Deepfake videos pose a significant threat in today’s digital age, making detection crucial. This project utilizes a 3D Convolutional Neural Network (3D CNN) to detect deepfake videos, leveraging temporal and spatial information in video frames for accurate classification.

The model is trained on the FaceForensics++ dataset and implemented using PyTorch, TensorFlow, and Keras.


**Features**

   > Detection of deepfake videos with high accuracy.

   > Utilizes 3D CNN architecture to analyze spatial and temporal features.

   > Preprocessed video frames for optimal input into the neural network.

   > Robust performance on the FaceForensics++ dataset.


**Technologies Used**


**Programming Languages:** Python


**Frameworks:** PyTorch, TensorFlow, Keras


**Dataset:** FaceForensics++


**Libraries:** OpenCV, NumPy, Scikit-learn


**Project Workflow**


**Data Preprocessing**

      > Extract frames from videos using OpenCV.
      
      > Normalize and resize frames for 3D CNN input.
      
      > Split data into training, validation, and test sets.

      
      
**Model Training**

      > Implemented a 3D CNN architecture to capture spatial and temporal features.
      
      > Used Cross-Entropy Loss as the loss function and Adam optimizer for training.
      
      > Regularized the model using Dropout to prevent overfitting.
      
**Evaluation**

      > Evaluated performance using metrics like accuracy, precision, recall, and F1-score.
      
      > Tested the model on FaceForensics++ to ensure robustness.

      
**Deployment**

      > Designed a pipeline for real-time deepfake detection using video streams.

      
**Model Architecture**

      > The 3D CNN architecture processes sequences of video frames (spatial data) and learns temporal relationships between them. Key layers include:

**3D Convolutional Layers:** Capture spatial and temporal features simultaneously.

**Batch Normalization:** Improves training speed and stability.

**ReLU Activation:** Introduces non-linearity.

**Fully Connected Layers:** For final classification.
