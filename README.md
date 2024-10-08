# GaitRecognition
CNN to identify walking patterns as an involuntary act of recognition

The project aimed to develop a Gait Recognition system capable of identifying individuals based on their unique walking patterns. The challenge was to leverage computer vision and machine learning techniques to process video data and recognize people based on their gait, using minimal data for training.

To solve this, we collected video recordings of five friends walking from left to right and converted the videos into silhouettes. The videos were broken down into individual frames, which were then used as inputs for a custom Convolutional Neural Network (CNN) model. The model was trained on these frames to extract distinguishing features, learning to identify each person based on their unique gait characteristics.

The outcome of this project was a model that achieved an F1-Score of 0.7, indicating a reasonable level of accuracy in predicting the identity of individuals from video frames. The system was evaluated by averaging predictions across multiple frames of a video to assign a class label to the person, demonstrating its effectiveness in recognizing gait patterns.

Skills and technologies used in this project included Python, TensorFlow, Keras, OpenCV, and computer vision techniques, all of which played a crucial role in data preprocessing, model training, and performance evaluation.
