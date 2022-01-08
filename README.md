<!-- PROJECT LOGO -->
<br />
<div align="center">
  <a href="https://github.com/commaai/speedchallenge">
    <img src="documentation/comma_logo.jpeg" alt="Logo" width="200" height="200">
  </a>
  <h1 align="left">Speed Challenge</h1>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
     <li>
      <a href="#data-analysis">Data Analysis</a>
        <ul>
            <li><a href="#input-data">Input Data</a></li>
            <li><a href="#histogram">Histogram</a></li>
        </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#approaches">Approaches</a></li>
        <ul>
            <li><a href="#optical-flow-approach">Optical Flow Approach</a></li>
            <li><a href="#time-series-analysis">Time Series Analysis</a></li>
        </ul>
      </ul>
    </li>
    <li><a href="#calculations">Calculations</a></li>
    <ul>
        <li><a href="#optical-flow-calculation">Optical Flow Calculation</a></li>
        <li><a href="#imbalance-correction">Imbalance Correction</a></li>
        <li><a href="#data-augmentation">Data Augmentation</a></li>
        <li><a href="#data-normalization">Data Normalization</a></li>
    </ul>
    <li><a href="#model-architecture">Model Architecture</a></li>
    <li><a href="#results">results</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://github.com/commaai)

This project is an interview challenge presented by comma.ai. The goal of this project is to predict the speed of the vehicle based on the Dash-Cam footage.

A huge thanks to Daniel Nugent, Jovan Sardinha, Othneil Drew and Ryan Chesler for sharing their solutions.

For someone like me starting out in AI and ML and not so proficient in python it was a huge help to walk through their solutions and grab the key ingredients.
## Data Analysis

### Input Data
One of the biggest challenges of the project was the limited availability of data for training. There was a single mp4 video with accompanying speed data. The video consists of 20400 frames and the speed vs frame graph is as follows:
![Input Speed Profile][speed-v-time]

The average speed is 12.18 and the speed varies from 30 to 0. Also we can see that from frame 7700 to 12100 there is a precipitous drop of velocity.

The input frames are of (480, 640, 3):

![Sample Frame][sample_frame]

### Histogram:
The distribution profile is:

![Histogram][histogram]

<!-- GETTING STARTED -->
## Getting Started:
In order to get started:
1. Get the <a href="https://github.com/nturumel/CarND-SpeedChallengeSimpleCNNNotebook">code</a>.
2. Setup docker.
3. Run `launch_docker.sh`.
4. Run the data `CarND Notebook - dataset generation` notebook.
5. Run either `CarND Notebook - training` or the `CarND Notebook - training -colab.ipynb` notebook.
### Approaches
#### Optical Flow Approach:
This approach is based on taking successive frames through an optical flow process. Open cv provides us with inbuilt methods to calculate the dense optical flow and sparse optical flow based on successive frames. Once we have this we can map the optical flow images to the speed data using a simple CNN or a deep learning model.
#### Time Series Analysis:
This approach is to enable the model to make connections across time that is not evident purely by looking at the optical flow data. We process individual frames through a CNN and then feed it into an LSTM and map it to a speed.

## Calculations
### Optical Flow Calculation
From <a href="https://www.sciencedirect.com/topics/engineering/optical-flow">Optical Flow</a>:
Optical flow is defined as the apparent motion of individual pixels on the image plane. It often serves as a good approximation of the true physical motion projected onto the image plane. Open cv provides two mechanisms for <a href="https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html">optical flow calculation</a>, Lucas-Kanade method and Dense Optical Flow in OpenCV.
We calculate the dense optical flow.

| [![Optical Flow][optical_flow]](https://learnopencv.com/optical-flow-in-opencv/) |
| :------------------------------------------------------------------------------: |
|                                  *Optical Flow*                                  |

| [![Lucaskanade Demo][lucaskanade-demo]](https://learnopencv.com/optical-flow-in-opencv/) |
| :--------------------------------------------------------------------------------------: |
|                                    *Lucaskanade Flow*                                    |

| [![Optical Flow Dense][optical-flow-dense]](https://learnopencv.com/optical-flow-in-opencv/) |
| :------------------------------------------------------------------------------------------: |
|                                     *Dense Optical Flow*                                     |



```python
def opticalFlowDense(image_current, image_next):
    """
    input: image_current, image_next (RGB images)
    calculates optical flow magnitude and angle and places it into HSV image
    * Set the saturation to the saturation value of image_next
    * Set the hue to the angles returned from computing the flow params
    * set the value to the magnitude returned from computing the flow params
    * Convert from HSV to RGB and return RGB image with same size as original image
    """
    gray_current = cv2.cvtColor(image_current, cv2.COLOR_RGB2GRAY)
    gray_next = cv2.cvtColor(image_next, cv2.COLOR_RGB2GRAY)
    
    
    hsv = np.zeros((66, 220, 3))
    # set saturation
    hsv[:,:,1] = cv2.cvtColor(image_next, cv2.COLOR_RGB2HSV)[:,:,1]
 
    # Flow Parameters
#     flow_mat = cv2.CV_32FC2
    flow_mat = None
    image_scale = 0.5
    nb_images = 1
    win_size = 15
    nb_iterations = 2
    deg_expansion = 5
    STD = 1.3
    extra = 0

    # obtain dense optical flow paramters
    flow = cv2.calcOpticalFlowFarneback(gray_current, gray_next,  
                                        flow_mat, 
                                        image_scale, 
                                        nb_images, 
                                        win_size, 
                                        nb_iterations, 
                                        deg_expansion, 
                                        STD, 
                                        0)
                                        
        
    # convert from cartesian to polar
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])  
        
    # hue corresponds to direction
    hsv[:,:,0] = ang * (180/ np.pi / 2)
    
    # value corresponds to magnitude
    hsv[:,:,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
    # convert HSV to float32's
    hsv = np.asarray(hsv, dtype= np.float32)
    rgb_flow = cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

    
    return rgb_flow
```
### Imbalance Correction: 
We need to correct the input imbalance and try to flatten the distribution:
```python
from sklearn.utils import resample

sample = train_meta_df[(train_meta_df.speed < 4) & (train_meta_df.speed > 3)]
res_df_1 = resample(sample, replace=True, n_samples=800, random_state=123)

sample = train_meta_df[(train_meta_df.speed < 12) & (train_meta_df.speed > 8)]
res_df_2 = resample(sample, replace=True, n_samples=1500, random_state=123)

sample = train_meta_df[(train_meta_df.speed < 14) & (train_meta_df.speed > 13)]
res_df_3 = resample(sample, replace=True, n_samples=1250, random_state=123)

sample = train_meta_df[(train_meta_df.speed < 19) & (train_meta_df.speed > 14.1)]
res_df_4 = resample(sample, replace=True, n_samples=2000, random_state=123)

sample = train_meta_df[(train_meta_df.speed < 30) & (train_meta_df.speed > 26)]
res_df_5 = resample(sample, replace=True, n_samples=2000, random_state=123)

res_df = pd.concat([train_meta_df_cp, res_df_1, res_df_2, res_df_3, res_df_4, res_df_5])
res_df.reset_index(inplace=True, drop=True) 
res_df.hist(column='speed', bins=10) 
```
![Imbalance Correction][imbalance_correction]
### Data Augmentation: 
We augment the data by changing the brightness input images. 
```python
def change_brightness(image, bright_factor):
    """
    Augments the brightness of the image by multiplying the saturation by a uniform random variable
    Input: image (RGB)
    returns: image with brightness augmentation
    """
    
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # perform brightness augmentation only on the second channel
    hsv_image[:,:,2] = hsv_image[:,:,2] * bright_factor
    
    # change back to RGB
    image_rgb = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    return image_rgb
```
### Data Normalization:
The input data is rescaled between -1 and 1:
```python
lambda_layer = Lambda(lambda x: x/ 127.5 - 1, input_shape = input_shape, name = 'lambda')(input)  
```

## Model Architecture
```python
input_shape = (N_img_height, N_img_width, N_img_channels)

    # normalization    
    # perform custom normalization before lambda layer in network
    input = Input(shape=input_shape, name = 'input')  
    lambda_layer = Lambda(lambda x: x/ 127.5 - 1, input_shape = input_shape, name = 'lambda')(input)    
    mobile_net = tf.keras.applications.MobileNetV2(input_shape=input_shape, alpha=1.0, include_top=False, weights="imagenet", input_tensor=None, pooling="avg")(lambda_layer)
    mobile_net.trainable = True
    flatten = Flatten(name = 'flatten')(mobile_net)
    x = Dense(600, activation='elu', name = 'fc1')(flatten)
    x = Dropout(0.3)(x)
    x = Dense(300, activation='elu', name = 'fc2')(x)
    x = Dropout(0.3)(x)
    x = Dense(150, activation='elu', name = 'fc3')(x)
    x = Dropout(0.3)(x)
    x = Dense(75, activation='elu', name = 'fc4')(x)
    x = Dropout(0.3)(x)
    x = Dense(15, activation='elu', name = 'fc5')(x)
    x = Dropout(0.25)(x)
    output = Dense(1, activation='elu', kernel_initializer = 'he_normal', name = 'output')(x)
    model = Model(inputs=input, outputs=output)


    model.compile(optimizer = adam, loss = 'mse')
```
## Results
Best: 
```
Epoch 126/150
100/100 [==============================] - ETA: 0s - loss: 7.6142
Epoch 00126: val_loss improved from 1.07929 to 1.06740, saving model to assets/model_assets/model=pretrained-batch_size=16-num_epoch=150-steps_per_epoch=100/weights.h5
100/100 [==============================] - 154s 2s/step - loss: 7.6142 - val_loss: 1.0674
```

| ![Training Validation Loss][training_validation_loss] |
| :------------------------------------------------------------------------------: |
|                                  *Training Validation Loss*                                  |
| ![Validation Fit][validation_fit] |
| :--------------------------------------------------------------------------------------: |
|                                    *Validation Fit*                                    |
| ![Result][result] |
| :--------------------------------------------------------------------------------------: |
|                                    *Result*                                    |

```
100%|██████████| 10796/10796 [11:51<00:00, 15.18it/s]
```
Results are stored in ```data/results.csv```
<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[product-screenshot]: documentation/logo.png 
[speed-v-time]: documentation/speed_v_time.png
[sample_frame]: documentation/sample_frame.png
[histogram]: documentation/histogram.png
[optical_flow]: documentation/optical_flow.png
[lucaskanade-demo]: documentation/lucaskanade-demo.png
[optical-flow-dense]: documentation/optical-flow-dense.jpg
[imbalance_correction]: documentation/imbalance_correction.png
[training_validation_loss]: documentation/training_validation_loss.png
[validation_fit]: documentation/validation_fit.png
[result]: documentation/result.gif
