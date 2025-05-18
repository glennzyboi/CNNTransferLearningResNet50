![image](https://github.com/user-attachments/assets/4961df36-0cb8-456c-a5ab-b58ae63007fb)# Facial Expression Recognition Through Transfer Learning: A Deep Learning Approach Using ResNet50

## Abstract
Our project tackles the task of emotion recognition from facial expressions using transfer learning with ResNet50. We implemented a CNN-based solution that classifies facial expressions into six emotional states: angry, fear, happy, neutral, sad, and surprise. We utilized the pre-trained ResNet50 architecture, fine-tuning it with a custom dataset to achieve optimal performance in emotion detection. Our approach incorporates progressive layer unfreezing, mixed-precision training, and data augmentation techniques. The model achieved significant improvements over the baseline, demonstrating the effectiveness of transfer learning in real-world computer vision applications.

## Task Definition & Dataset

### Task Overview
- **Classification Type**: Multi-class classification (6 classes)
- **Input**: Facial images
- **Output**: Emotional state prediction
- **Classes**: angry, fear, happy, neutral, sad, surprise

### Dataset Details
- **Source**: Face expression recognition dataset 
(https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
- **Structure**:
  - Training set: Balanced distribution across emotions
  - Validation set: 20% of total data
  - Image dimensions: 224x224 pixels (ResNet50 standard input size)
- **Preprocessing**:
  - Image resizing and normalization
  - Data augmentation (random flips, rotations, brightness adjustments)
  - Class balancing through weighted sampling

## Model & Methodology

### Base Model Selection
We chose ResNet50 as our base model for several reasons:
- Pre-trained on ImageNet, providing robust feature extraction capabilities
- Deep architecture with residual connections, preventing vanishing gradients
- Proven track record in transfer learning tasks
- Efficient training with relatively lower computational requirements

### Fine-tuning Approach
1. **Initial Phase**:
   - Froze all ResNet50 layers
   - Added custom classifier layers with dropout for regularization
   - Trained only the new layers

2. **Progressive Unfreezing**:
   - Unfroze layer3 for fine-grained feature adaptation
   - Implemented gradual learning rate adjustment
   - Used mixed-precision training for efficiency

### Technical Configuration
- **Framework**: PyTorch
- **Training Parameters**:
  - Batch size: 64
  - Base learning rate: 1e-3
  - Weight decay: 1e-4
  - Early stopping patience: 3 epochs
  - Maximum epochs: 20
- **Optimizer**: AdamW with OneCycleLR scheduler
- **Loss Function**: Cross-entropy with label smoothing (0.1)

### Compute Environment
- Development platform: VSCode
- Hardware: NVIDIA GPU (when available)
- Mixed precision training enabled for optimal GPU utilization

## Qualitative Analysis

### Model Performance
- Successfully captures subtle emotional expressions
- Robust to varying lighting conditions and facial orientations
- Improved accuracy on ambiguous expressions after fine-tuning

### Key Observations
1. **Improvements**:
   - Better distinction between similar emotions (e.g., fear vs. surprise)
   - Reduced false positives in neutral expression detection
   - More robust to partial facial occlusions

2. **Challenges**:
   - Occasional confusion between sad and neutral expressions
   - Performance variation with extreme head poses
   - Sensitivity to image quality

## Conclusion

### Learning Outcomes
1. **Technical Skills**:
   - Practical implementation of transfer learning
   - Understanding of progressive layer unfreezing
   - Experience with PyTorch and GPU acceleration

2. **Model Performance**:
   - Significant improvement over baseline ResNet50
   - Successful adaptation to emotion recognition task
   - Practical applicability demonstrated
  
![image](https://github.com/user-attachments/assets/050a4062-7bc1-40f5-b17a-95dee42a6783)

![image](https://github.com/user-attachments/assets/c0a712d4-6a8f-4517-b466-5db0180118c2)

![image](https://github.com/user-attachments/assets/5a853483-0eec-4c07-8472-09d1f7cbd51c)

![image](https://github.com/user-attachments/assets/eeb83cdd-441d-4ef8-9001-575fda5bd2d7)

Classification Report:
              precision    recall  f1-score   support

       angry     0.5118    0.6323    0.5657       960
        fear     0.5770    0.2760    0.3734      1018
       happy     0.8615    0.7874    0.8228      1209
     neutral     0.5770    0.6472    0.6101      1216
         sad     0.4868    0.5812    0.5298      1139
    surprise     0.7264    0.7629    0.7442       797

    accuracy                         0.6148      6339
   macro avg     0.6234    0.6145    0.6077      6339
weighted avg     0.6240    0.6148    0.6084      6339

Overall Accuracy: 0.6148
Validation Accuracy: 0.6148

![image](https://github.com/user-attachments/assets/40eaa1b5-c122-4ac1-82f3-885acdb0a7cd)

Comparison w/ Base model

![image](https://github.com/user-attachments/assets/c39155dd-24a5-4bba-96c6-bb56badcf60c)


### Future Improvements
- Experiment with other architectures (EfficientNet, Vision Transformer)
- Implement attention mechanisms
- Expand dataset with more diverse samples

## Team Members
- Jhon Glenn L. Fabul
- Sid Digamon
- James Manon-og
- Emmannuel Aguado
- Adolfo Cedric Sarillo

## Repository Structure
```
.
├── EmotionRecognition.ipynb    # Main notebook with implementation
├── README.md                   # Project documentation
└── requirements.txt            # Dependencies
```

## Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- CUDA-capable GPU (recommended)

### Installation
```bash
git clone https://github.com/glennzyboi/CNNTransferLearningResNet50.git
cd CNNTransferLearningResNet50
pip install -r requirements.txt
```

### Usage
1. Open `EmotionRecognition.ipynb` in Jupyter Notebook, Google Colab, or VSCode
2. Follow the notebook cells for step-by-step execution
3. Ensure GPU runtime is enabled for optimal performance

## Acknowledgments
- Google Machine Learning Crash Course for foundational concepts
- PyTorch documentation and community
- Course instructor Kuya Soy Goated
