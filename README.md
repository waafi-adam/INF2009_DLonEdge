**Real-time Inference of Deep Learning models on Edge Device**

**Objective:** By the end of this session, participants will understand 
1. How to run a deep learning model on an edge device (aka raspberryPi) using Pytorch and RTLite Libraries
2. Deploy and run quantized version of a Large Language Model
3. [OPtional] How does quantization work and different quantization methods for deep learning models 

---

**Prerequisites:**
1. Raspberry Pi with Raspbian OS installed.
2. MicroSD card (16GB or more recommended).
3. Web camera compatible with Raspberry Pi.
4. Internet connectivity (Wi-Fi or Ethernet).
5. Basic knowledge of Python and Linux commands.

---

**1. Introduction**
- Edge analytics with real-time processing capabilities is chellenging but important and inevitable due to privacy/security concerns. However, edge devices like RaspberryPi are constrained with limited hardware resources, which at times are not sufficient to run complex deep learning models. These models require lot of computational resource and memory due to their size and complex architecture. Therefore, in such scenarios, we optimize the model such that it can run efficiently with reduced inference time critical for real-time analytics. Optimization can be achieved by combination of techniques like quantization and converting trained model into architecture specific lite model. 

**2. Running Deep Learning Model On RaspberryPi**
- **This section guide you on how to setup a Raspberry Pi for running PyTorch and deploy a MobileNet v2 image classification model in real time on the CPU.**
-  Set up and activate a virtual environment named "dlonedge" for this experiment (to avoid conflicts in libraries) as below.
  ```bash
  sudo apt install python3-venv
  python3 -m venv dlonedge
  source dlonedge/bin/activate
  ```

- Installing PyTorch and OpenCV:
  ```bash
  pip install torch torchvision torchaudio
  pip install opencv-python
  pip install numpy --upgrade
  ```

- Same as last lab, for video capture we’re going to be using OpenCV to stream the video frames. The model we are going to use in this lab is MobileNetV2, which takes in image sizes of 224x224. We are targeting 30fps for the model but we will request a slightly higher framerate of 36 fps than that so there is always enough frames and bandwidth of image pre-processing and model prediction.

- **Part 1.** [sample code](Codes/mobile_net.py) is used to directly load pre-trained MobileNetV2 model, doing model inference and finally, Observe the fps as shown in screenshot below when run on RaspberryPi 4B. As shown, with no optimization of model, we could only achieve of 5-6 fps much below our desired target.

  ![image1](https://github.com/user-attachments/assets/8e3cf302-45f3-41c9-85a5-a1bd118d30c4)

- **Part 2.** Edit line number 11 as shown below to enable quantization in [sample code](Codes/mobile_net.py) to use quantized version of MobileNetV2 model.

  ```bash
quantize = True
  ```

Finally, observe the fps as shown in screenshot below after using quantized model of MobileNetV2. We can now achieve close to 30 fps as required because of smaller footprint of quantized model.

  ![image2](https://github.com/user-attachments/assets/7086f300-4edf-4c41-a799-c496001ee1d1)

[Quantization](https://pytorch.org/docs/stable/quantization.html) techniques enable computations and tensor storage at reduced bitwidths compared to floating-point precision. In a quantized model, some or all operations use this lower precision, resulting in a smaller model size and the ability to leverage hardware-accelerated vector operations.

- **Part 3.** Uncomment lines 57-61 in [sample code](Codes/mobile_net.py) to print the top 10 predictions in real-time as shown in below video.

https://github.com/user-attachments/assets/5ee2a4c8-1988-4021-b194-aa0786a1ebfc

