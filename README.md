# YOLOv3-ReCaptcha

A proof of concept Recaptcha solver using YOLOv3 on Tensorflow 2.0 and Selenium. This tutorial shows that with a better trained object detection weight file, ReCaptcha can be easily solved.

## Installation

Install requirements and download pretrained weights:

```
$ pip3 install -r ./docs/requirements.txt
$ wget https://pjreddie.com/media/files/yolov3.weights
```
Download [ChromeDriver](https://sites.google.com/a/chromium.org/chromedriver/downloads) and put it into $PATH.


## Quick start

In this part, we will use pretrained weights to make predictions on ReCaptcha.

```
$ python selenium_demo.py
```

<p align="center">
    <img width="100%" src="https://ozankaraali.com/images/yolo-example1.png" style="max-width:100%;">
    </a>
</p>
<p align="center">
    <img width="100%" src="https://ozankaraali.com/images/yolo-example2.png" style="max-width:100%;">
    </a>
</p>
<p align="center">
    <img width="100%" src="https://ozankaraali.com/images/yolo-example3.png" style="max-width:100%;">
    </a>
</p>

## TODO:

- Click images using its pixel positions after finding objects and click VERIFY.
- Using NLP to find relation between "find buses" -> "bus" etc.
- Solve multiple screens such as "skip" or "until no objects left"

## Credits:
[YunYang1994](https://github.com/YunYang1994/TensorFlow2.0-Examples), [Igor Savinkin](https://stackoverflow.com/questions/32249190/improve-recaptcha-2-0-solving-automation-script-selenium/32415298), [Joseph Chet Redmon](https://pjreddie.com/darknet/yolo/)