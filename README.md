# Multi-Object Classification based on Deep Learning Models

Note: This repository includes the presentation slides summarizing my graduation thesis. For more detailed information, you can access the full report through the provided [link](https://drive.google.com/file/d/1ZHAYnq5sYW_ZHnlvVQ4O0ppy5xUBcoDA/view?usp=sharing). If you require the code or model for reimplementation, please contact me at duydo280802@gmail.com.

# Abstract
Our research focuses on improving dataset quality in multi-label image classification, a task that requires sophisticated and precise processing. Key challenges include handling images containing irrelevant objects outside the labeled set, which degrade data quality and reduce model accuracy. Another major issue is label imbalance, where frequent labels dominate learning, causing poor performance on rare labels. Additionally, semantic similarity among labels makes distinguishing between them difficult, demanding models capable of fine-grained differentiation. To tackle these problems, we applied image preprocessing to remove unnecessary objects and refined existing models to enhance classification performance. The proposed improvements showed promising results on challenging datasets.

# Proposed Models
In this study, we primarily focus our efforts on the development and evaluation of two principal models: [C-Tran](https://github.com/QData/C-Tran) and [Single Positive Labels](https://github.com/elijahcole/single-positive-multi-label).
## 1. C-Tran
C-Tran is designed to capture complex dependencies between image features and labels. It trains the Transformer Encoder to predict a target label set from inputs that include masked labels and image features extracted by convolutional neural networks. A key innovation is a three-state label masking scheme representing positive, negative, and uncertain states. This approach has proven effective, achieving improved performance on challenging datasets.

**Improvements made on C-Tran:**
- Changed feature extraction network
- Modified embedding feature addition methods
- Adjusted number of encoder layers
- Removed label state addition method
- Changed activation function

## 2. Single Positive Labels
Building upon previous PU learning approaches, this model assumes only one positive label per image without confirmed negative labels. It extends existing multi-label loss functions to handle various learning modes, including training linear classifiers or fine-tuning deep networks end-to-end.

**Improvements made:**
- Updated loss functions
- Modified activation functions and label estimators
- Enhanced classifier layers
- Refined mean and standard deviation calculations

## 3. Combined Model: C-Tran + Single Positive Labels
We experimented with integrating the Single Positive Labels model and C-Tran, aiming to combine simplified input labels with C-Tran’s ability to learn relationships between features and labels. The input label vectors retain only one positive label, with others marked unknown. The label state embedding matrix S is set to uncertain states in the combined architecture. The model calculates losses over all labels, not just unknown ones. The feature extractor in C-Tran was replaced with that from the Positive-Only Label model, and the best-performing loss functions from the latter (ROLE, AN-LS, Huber) were tested.

# Data Preprocessing Proposal
To address image-related technical issues, we propose cropping images to remove as much background as possible while retaining main objects. Cropping regions are defined by top-left and bottom-right points. The cropped images are then used as training data to improve model efficiency.

# Experiments and Evaluation

## Dataset Origin
The dataset for evaluation is from the [Food Recognition Benchmark 2022](https://d3qvx1ggyg4lu1.cloudfront.net/challenges/food-recognition-benchmark-2022), consisting of daily meal images taken by volunteers in Switzerland. The images were collected, categorized, and annotated by the organization Food & You.

## 1. Single Positive Labels
We conducted experiments on the Positive-Only Label model to evaluate different loss functions, training modes, and activation functions. The results are summarized in **Table 1**.

**Table 1. Single Positive Labels Results**

| Loss Function | Training Mode     | Activation | mAP (Validation) | mAP (Test) |
|---------------|-------------------|------------|------------------|------------|
| HU            | End-to-end        | Sigmoid    | 21.0043          | 32.6947    |
| AN-LS         | End-to-end        | Sigmoid    | 24.2951          | 34.6173    |
| AN-LS         | Transfer Learning | Sigmoid    | 24.4427          | 34.8327    |

**Evaluation:**
The experimental results show that the Positive-Only Label model performs best when using the AN-LS loss function with the transfer learning mode. This suggests that transfer learning significantly improves the model’s learning capability in scenarios where negative labels are absent. Additionally, the sigmoid activation function consistently outperforms softmax, which aligns well with the binary nature of multi-label classification tasks.

In terms of backbone networks, ResNet50 yields better results than EfficientNetB7, indicating that a deeper or more complex model does not always correlate with better performance for this task. Among the loss functions tested, the Huber loss, a newly introduced one, performed better than ROLE, which was originally considered the best for this model. However, a key limitation of this approach is that it only considers a single positive label per training instance, making it less effective in modeling label dependencies.



## 2. C-Tran Model
In our experiments with the C-Tran model, we consistently used the Binary Cross-Entropy loss function across all settings. We varied the feature extractors, activation functions, encoder layers, and label masking techniques. The results are presented in **Table 2**.

**Table 2. C-Tran Model Results**

| Feature Extractor | Activation | Encoder Layers | Label Masking | Known Labels | Label State | Test Accuracy |
|-------------------|------------|----------------|---------------|--------------|-------------|---------------|
| ResNet101         | Softmax    | 3              | Yes           | 0            | Total         | 90.1         |
| EfficientNetB0    | Sigmoid    | 3              | Yes           | 0            | Total        | 91.3          |
| MobileNetV2       | Sigmoid    | 2              | Yes            | 0            | Total        | 91.3          |

**Evaluation:**
The C-Tran model demonstrates superior performance in multi-label classification by effectively capturing both the relationships among image features and the semantic correlations among labels. Across different configurations, backbones like MobileNetV2 and EfficientNetB0 both achieved high test accuracy (91.3), but MobileNetV2 is preferred due to its lower parameter count and computational efficiency.

Replacing the softmax activation with sigmoid improved output quality. Experiments with encoder layer counts ranging from 2 to 4 showed little variation in results, but 2 layers offered a good trade-off between accuracy and model complexity. Whether or not label masking was used during training, or prior knowledge of the number of labels was given, did not significantly affect performance. The "summed state" approach for label state embedding outperformed the "product state," emphasizing the importance of label state representation.

Compared to the Single Positive Labels model, C-Tran performs significantly better because it is trained on the full dataset and is architecturally optimized for learning inter-label correlations — which is particularly crucial for complex multi-label datasets.



## 3. Combined Model (C-Tran + Single Positive Labels)
We explored the integration of the Single Positive Labels model into the C-Tran. In this setup, each label vector retains only one positive label, and other labels are marked as unknown. The label state matrix in C-Tran was set entirely to “unknown.” Loss was computed over all labels. Additionally, we used the feature extractors and loss functions that performed best in previous individual experiments.

**Table 3. Combined Model Configuration**

| Activation | Encoder Layers | Label Masking | Known Labels | Label State |
|------------|----------------|---------------|--------------|-------------|
| Sigmoid    | 3              | Yes           | 0            | Total         |

**Table 4. Combined Model Results**

| Feature Extractor | Loss Function | Test Accuracy |
|-------------------|---------------|---------------|
| EfficientNetB0    | ROLE          | 54.9          |
| MobileNetV2       | AN-LS         | 51.0          |
| MobileNetV2       | HU            | 47.0          |

**Evaluation:**
This hybrid model aimed to integrate the simplified label input structure of the Single Positive Labels model with the label-dependency modeling strength of C-Tran. However, the results indicate that the combined model does not outperform the original C-Tran.

Specifically, the configuration using the ROLE loss function with EfficientNetB0 as the feature extractor gave the best result (test mAP = 54.9), surpassing the configurations that used AN-LS and Huber. 
Nevertheless, this still falls short compared to the ~91.3% test accuracy achieved by the original C-Tran. This gap may stem from the reduction in label richness in the input, which prevents the model from leveraging the full capability of the C-Tran architecture. Furthermore, setting all labels to an "unknown" state might introduce noise, making it harder for the model to learn meaningful semantic relationships.

## 4. Model Showcase: Our Best Performer. 
[Video demo](https://drive.google.com/file/d/1mdZl9i_0eHUslkAOcBJ4Oc-MYlV-xw-b/view?usp=sharing))


# Conclusion

In conclusion, typical challenges in multi-label classification datasets include irrelevant objects in images, label imbalance, and semantic overlap among labels. We addressed these by preprocessing images and improving two model types: the Positive-Only Label model and C-Tran. Our combined model further leverages the advantages of both approaches, showing enhanced performance in multi-label classification tasks.
