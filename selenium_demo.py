from time import sleep
from random import uniform
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.support import expected_conditions as EC

import PIL
import cv2
import numpy as np
import core.utils as utils
import tensorflow as tf
from core.yolov3 import YOLOv3, decode
from PIL import Image


def hover(element):
    hov = ActionChains(driver).move_to_element(element)
    hov.perform()


driver = webdriver.Chrome()
driver.get("https://www.google.com/recaptcha/api2/demo")

# Credit: https://stackoverflow.com/questions/32249190/improve-recaptcha-2-0-solving-automation-script-selenium/32415298

recaptchaFrame = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, 'iframe')))
frameName = recaptchaFrame.get_attribute('name')
driver.switch_to.frame(frameName)
CheckBox = WebDriverWait(driver, 10).until(
    EC.presence_of_element_located((By.ID, "recaptcha-anchor"))
)

rand = uniform(1.0, 1.5)
print('\n\r explicit wait for ', rand, ' seconds...')
sleep(rand)
hover(CheckBox)

rand = uniform(0.5, 0.7)
print('\n\r explicit wait for ', rand, 'seconds...')
sleep(rand)
clickReturn = CheckBox.click()
print('\n\r after click on CheckBox... \n\r CheckBox click result: ', clickReturn)

driver.switch_to.parent_frame()
recaptchaFrame = WebDriverWait(driver, 10).until(EC.presence_of_all_elements_located((By.TAG_NAME, 'iframe')))
frameName = recaptchaFrame[1].get_attribute('name')

rand = uniform(1, 3)
print('\n\r explicit wait for ', rand, 'seconds...')
sleep(rand)

pngimg = recaptchaFrame[1].screenshot_as_png
f = open('file.png', 'wb')
f.write(pngimg)
f.close()

# Credit: https://github.com/YunYang1994/TensorFlow2.0-Examples/tree/master/4-Object_Detection/YOLOV3

input_size = 416
image_path = "file.png"

input_layer = tf.keras.layers.Input([input_size, input_size, 3])
feature_maps = YOLOv3(input_layer)

original_image = cv2.imread(image_path)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
original_image_size = original_image.shape[:2]

image_data = utils.image_preporcess(np.copy(original_image), [input_size, input_size])
image_data = image_data[np.newaxis, ...].astype(np.float32)

bbox_tensors = []
for i, fm in enumerate(feature_maps):
    bbox_tensor = decode(fm, i)
    bbox_tensors.append(bbox_tensor)

model = tf.keras.Model(input_layer, bbox_tensors)
utils.load_weights(model, "./yolov3.weights")
model.summary()

pred_bbox = model.predict(image_data)
pred_bbox = [tf.reshape(x, (-1, tf.shape(x)[-1])) for x in pred_bbox]
pred_bbox = tf.concat(pred_bbox, axis=0)
bboxes = utils.postprocess_boxes(pred_bbox, original_image_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')

image = utils.draw_bbox(original_image, bboxes)
image = Image.fromarray(image)
image.show()

driver.close()