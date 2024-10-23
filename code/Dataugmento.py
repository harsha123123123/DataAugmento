import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt

def show_image(img_path):
  # Load the image using PIL
  image = Image.open(image_path)

  # Display the image using matplotlib
  fig, ax = plt.subplots()
  ax.imshow(image)
  ax.axis("off")
  st.pyplot(fig)


# Set Page
st.set_page_config(page_title = "Dataugmento")
st.markdown("\n")
st.markdown("\n")

# Title Style -  Welcome to Dataugmento!

original_title = '<p style="font-family:Courier; color:orange; font-size: 30px;font-weight: bold;"> Welcome to Dataugmento!</p>'
st.markdown(original_title, unsafe_allow_html=True)
st.markdown("""<style>p {font-size: 24px;}</style>""",unsafe_allow_html=True)
st.markdown("---")

# What is Dataugmento?
st.write("What is Dataugmento?")
st.text('Dataugmento converts the users uploaded small image data to a large image dataset \nwhich can be used to train various ML models.')
st.text("\n")

# Tutorial Heading
st.markdown("""<style>p {font-size: 24px;}</style>""",unsafe_allow_html=True)
st.write("Here is quick tutorial to use our tool:")

# Video
video_file = open("Dataugmento_demo.mp4", "rb").read()
st.video(video_file)

# Methods Heading
st.markdown("""<style>p {font-size: 24px;}</style>""",unsafe_allow_html=True)
st.markdown("---")
st.write("Data Augmentation methods available in this tool are:")
st.text("Below is a quick description about each method that is available. You can choose \nmethods based on your dataset.")
st.markdown("\n")
# Cropping
st.markdown("<p style='font-family: Garamond;'> Cropping </p>", unsafe_allow_html=True)
st.text("\nIn this method the it selects a random region of the image, \nand generates that cropped part as a new image.")
image_path = "cropping.png"
show_image(image_path)

# Rotation
st.markdown("<p style='font-family: Garamond;'> Rotation </p>", unsafe_allow_html=True)
st.text("\nIn this method the source image is random rotated clockwise or \ncounterclockwise by some number of degrees, changing the position of the object\nin frame.")
image_path = 'Rotation.png'
show_image(image_path)
st.markdown("\n")

# Flipping
st.markdown("<p style='font-family: Garamond;'>  Flipping </p>", unsafe_allow_html=True)
st.text("\nFlipping is a data augmentation technique in which we rotate an image in a horizontal or vertical axis.\nIn horizontal flip, the flipping will be on vertical axis, In Vertical flip \nthe flipping will be on horizontal axis.")
image_path = "Flipping.png"
show_image(image_path)
st.markdown("\n")

# Mix-up
st.markdown("<p style='font-family: Garamond;'> Mix-up </p>", unsafe_allow_html=True)
st.text(" \nMixup is a data augmentation technique that generates a weighted combination \nof random image pairs from the training data.")
image_path = "mixup2.0.png"
show_image(image_path)
st.markdown("\n")


# Kernel Filtering 
st.markdown("<p style='font-family: Garamond;'> Kernel Filtering </p>", unsafe_allow_html=True)
st.text("\nKernel Filter is a type of smoothing method which is often applied to an input \nvector, time series or matrix to generate the smoothed version of the input \nsequence.")
image_path = 'kernel_filter.png'
show_image(image_path)
st.markdown("\n")

# Random Erasing
st.markdown("<p style='font-family: Garamond;'> Random Erasing </p>", unsafe_allow_html=True)
st.text("\nRandom Erasing is a data augmentation method, which randomly selects a \nrectangle region in an image and erases its pixels with random values.")
image_path =  "random_erasing.png"
show_image(image_path)
st.markdown("\n")

# Grey Scaling
st.markdown("<p style='font-family: Garamond;'> Grey Scaling  </p>", unsafe_allow_html=True)
st.text("\nGrayscaling is the process of converting an image i.e. RGB to another color spaces \ni.e. gray.")
image_path =  "Gray Scaling.png"
show_image(image_path)

# Colour - Space transformation
st.markdown('\n')
st.markdown("<p style='font-family: Garamond;'>Colour - Space transformation </p>", unsafe_allow_html=True)
st.text('Color space transformation is a type of data augmentation technique that involves \nconverting an image from one color space to another.')
image_path = 'color_transformation.png'
show_image(image_path)

# Noise Injection
st.markdown('\n')
st.markdown("<p style='font-family: Garamond;'> Noise Injection  </p>", unsafe_allow_html=True)
st.text("\nNoise injection is a type of data augmentation technique used to increase the \ndiversity of a dataset by adding noise to the input data.")
image_path = 'Noise.png'
show_image(image_path)

# hide_st_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}header {visibility: hidden;}</style>"""
# st.markdown(hide_st_style, unsafe_allow_html=True)