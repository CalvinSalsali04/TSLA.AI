🚀 **TSLA.AI: Custom LLM for Tesla Stock Prediction**
=====================================================

### *Fine-tuned from Scratch to Predict Tesla Stock Movements*

**Overview**
---------------

**TSLA.AI** is a custom-built Large Language Model (LLM) designed to predict Tesla stock trends by analyzing Elon Musk's tweets and Tesla-related news articles. This project leverages a GPT model developed from scratch, fine-tuned for sentiment analysis, and automated to send daily investment predictions via SMS.

Currently sitting at 9.35% in profit, will be updating my % and accuracy as the model predicts more.

*Example of the model texting me on 2025-01-03 to buy Tesla stock!*
![image](https://github.com/user-attachments/assets/e1927f51-ae75-4682-adbc-7dbf9374c7b9)
<img width="444" alt="image" src="https://github.com/user-attachments/assets/763eee35-d545-4d0d-aac4-6313da7d35cb" />
<img width="1033" alt="image" src="https://github.com/user-attachments/assets/d2abe0c7-4be3-4134-84d4-7492ab033ceb" />



* * * * *

⚙️ **How TSLA.AI Works**
------------------------

1.  **Real-time Data Collection**

    -   Scrapes Tesla-related news and Elon Musk's tweets using Tweepy and NewsAPI.
    -   Aggregates stock data for sentiment-based predictions.
2.  **Custom GPT Model**

    -   GPT model built from scratch using PyTorch.
    -   Pre-trained on financial datasets and fine-tuned with Tesla-specific content.
3.  **Prediction Pipeline**

    -   Predicts "Buy", "Sell", or "Hold" daily based on sentiment analysis.
    -   Sends predictions directly to my phone via Twilio SMS.

* * * * *

🚀 **Features**
---------------

-   **🛠️ GPT Model from Scratch** -- Full transformer architecture built manually.
-   **📊 Tesla-Specific Finetuning** -- Trained on Elon Musk's tweets and Tesla-related articles.
-   **📱 SMS Predictions** -- Receive daily investment recommendations via Twilio.
-   **🔒 Dynamic Attention Masking** -- Improves accuracy and stability during prediction.
-   **📈 Continuous Fine-Tuning** -- Periodically retrained on updated datasets for better performance.

* * * * *

🧪 **Training Process**
-----------------------

### Pre-Training (Financial Data)

bash

Copy code

`~/Desktop/llmfromscratch/src master ❯ python3 training/train.py
Epoch 1/10, Loss: 4.7427
Epoch 2/10, Loss: 3.6742
Epoch 3/10, Loss: 3.2924
Epoch 4/10, Loss: 3.0384
Epoch 5/10, Loss: 2.8566
Epoch 6/10, Loss: 2.7204
Epoch 7/10, Loss: 2.6172
Epoch 8/10, Loss: 2.5275
Epoch 9/10, Loss: 2.4596
Epoch 10/10, Loss: 2.4006
Training complete!`

### Fine-Tuning on Tesla Data

bash

Copy code

`~/Desktop/llmfromscratch/src master ❯ python3 training/finetune.py
Loaded pre-trained weights.
Epoch 1/15, Loss: 3334.4570
Validation Loss: 530.4249, Validation Accuracy: 8.59%
Epoch 2/15, Loss: 2772.8207
Validation Loss: 285.9842, Validation Accuracy: 39.62%
Epoch 3/15, Loss: 2625.1603
Validation Loss: 465.7320, Validation Accuracy: 42.00%
Epoch 4/15, Loss: 2560.5375
Validation Loss: 372.7053, Validation Accuracy: 42.72%
Epoch 15/15, Loss: 2402.8423
Validation Loss: 445.0124, Validation Accuracy: 43.44%
Fine-tuning complete.`

* * * * *

📊 **Fine-Tuning Reflection**
-----------------------------

### **Challenges Faced**

1.  **Domain Shift**

    -   Pre-training focused on general financial news, while fine-tuning involved Tesla-specific tweets. This discrepancy led to fluctuating accuracy during early fine-tuning epochs.
2.  **Overfitting**

    -   The model's training loss consistently dropped, but validation accuracy plateaued around **43.44%** after Epoch 6. This suggested overfitting to the training data.
3.  **Volatile Early Results**

    -   Initial validation accuracy was **8.59%** at Epoch 1, jumping to **39.62%** by Epoch 2. The rapid spike indicated early learning but highlighted the need for further regularization.

* * * * *

### **Steps Taken to Improve Performance**

-   **Learning Rate Scheduler** -- Reduced learning rate by **40% every epoch** to fine-tune incrementally.
-   **Gradient Clipping** -- Applied `clip_grad_norm_` to avoid exploding gradients.
-   **Dropout Adjustment** -- Increased dropout rate from **0.1 to 0.2** for added regularization.
-   **Weight Decay** -- Added `weight_decay=0.02` to prevent overfitting.

* * * * *

### **Results**

-   **Final Validation Accuracy**: **43.44%**
-   **Training Loss**: Reduced from **3334.46** to **2402.84**
-   **Validation Loss Plateau**: Around **445.01** by the end of training.

* * * * *

🔮 **Future Improvements**
--------------------------

1.  **Data Augmentation**

    -   Expand dataset by incorporating more Tesla-specific tweets and news articles.
2.  **Ensemble Models**

    -   Combine predictions from **TSLA.AI** with Cohere-based sentiment classifiers for more robust results.
3.  **Masked Language Modeling**

    -   Randomly mask **10% of tokens** during training to enhance contextual learning.
4.  **Larger Dataset**

    -   Increase the dataset size to further reduce overfitting and improve generalization.

* * * * *

🛠️ **Tech Stack**
------------------

-   **Python** -- Primary development language.
-   **PyTorch** -- Model development and training.
-   **Tweepy & NewsAPI** -- Real-time data collection.
-   **Twilio** -- SMS integration for delivering predictions.
-   **Cohere** -- Advanced sentiment analysis for additional insights.
-   **Pandas** -- Data manipulation and preprocessing.
-   **dotenv** -- Secure API key management.

* * * * *
