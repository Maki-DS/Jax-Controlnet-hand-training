# Jax-Controlnet-hand-training
Training a controlnet using mediapipe landmarks with huggingface models and datasets.

<h2><b>Summary 📋</b></h2>

As Stable diffusion and other diffusion models are notoriously poor at generating realistic hands for our project we decided to train a ControlNet model using MediaPipes landmarks in order to generate more realistic hands avoiding common issues such as unrealistic positions and irregular digits.
        <br>
        We opted to use the [HAnd Gesture Recognition Image Dataset](https://github.com/hukenovs/hagrid) (HaGRID) and [MediaPipe's Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) to train a control net that could potentially be used independently or as an in-painting tool.
        To preprocess the data there were three options we considered:
        <ul>
            <li>The first was to use Mediapipes built-in draw landmarks function. This was an obvious first choice however we noticed with low training steps that the model couldn't easily distinguish handedness and would often generate the wrong hand for the conditioning image.</li>
        <center>
        <table><tr>
        <td> 
          <p align="center" style="padding: 10px">
            <img alt="Forwarding" src="https://datasets-server.huggingface.co/assets/MakiPan/hagrid250k-blip2/--/MakiPan--hagrid250k-blip2/train/29/image/image.jpg" width="200">
            <br>
            <em style="color: grey">Original Image</em>
          </p> 
        </td>
        <td> 
          <p align="center">
            <img alt="Routing" src="https://datasets-server.huggingface.co/assets/MakiPan/hagrid250k-blip2/--/MakiPan--hagrid250k-blip2/train/29/conditioning_image/image.jpg" width="200">
            <br>
            <em style="color: grey">Conditioning Image</em>
          </p> 
        </td>
        </tr></table>
        </center>
            <li>To counter this issue we changed the palm landmark colors with the intention to keep the color similar in order to learn that they provide similar information, but different to make the model know which hands were left or right.</li>   
        <center>
        <table><tr>
        <td> 
          <p align="center" style="padding: 10px">
            <img alt="Forwarding" src="https://datasets-server.huggingface.co/assets/MakiPan/hagrid-hand-enc-250k/--/MakiPan--hagrid-hand-enc-250k/train/96/image/image.jpg" width="200">
            <br>
            <em style="color: grey">Original Image</em>
          </p> 
        </td>
        <td> 
          <p align="center">
            <img alt="Routing" src="https://datasets-server.huggingface.co/assets/MakiPan/hagrid-hand-enc-250k/--/MakiPan--hagrid-hand-enc-250k/train/96/conditioning_image/image.jpg" width="200">
            <br>
            <em style="color: grey">Conditioning Image</em>
          </p> 
        </td>
        </tr></table>
        </center>
            <li>The last option was to use <a href="https://ai.googleblog.com/2020/12/mediapipe-holistic-simultaneous-face.html">MediaPipe Holistic</a> to provide pose face and hand landmarks to the ControlNet. This method was promising in theory, however, the HaGRID dataset was not suitable for this method as the Holistic model performs poorly with partial body and obscurely cropped images.</li>
        </ul>
        We anecdotally determined that when trained at lower steps the encoded hand model performed better than the standard MediaPipe model due to implied handedness. We theorize that with a larger dataset of more full-body hand and pose classifications, Holistic landmarks will provide the best images in the future however for the moment the hand-encoded model performs best.

<h2><b>Links 🔗</b></h2>

### Models 🚀
<h4><a href="https://huggingface.co/Vincent-luo/controlnet-hands">Standard Model</a></h4>
<h4> <a href="https://huggingface.co/MakiPan/controlnet-encoded-hands-130k/">Model using Hand Encoding</a></h4>


### Datasets 💾
<h4> <a href="https://huggingface.co/datasets/MakiPan/hagrid250k-blip2">Dataset for Standard Model</a></h4>
<h4> <a href="https://huggingface.co/datasets/MakiPan/hagrid-hand-enc-250k">Dataset for Hand Encoding Model</a></h4>


### Preprocessing Scripts 📑

<h4> <a href="https://github.com/Maki-DS/Jax-Controlnet-hand-training/blob/main/normal-preprocessing.py">Standard Data Preprocessing Script</a></h4>
<h4> <a href="https://github.com/Maki-DS/Jax-Controlnet-hand-training/blob/main/Hand-encoded-preprocessing.py">Hand Encoding Data Preprocessing Script</a></h4></center>
            
