import streamlit as st
import cv2
from PIL import Image
import random
import os
import numpy as np
from streamlit_option_menu import option_menu
import os
import shutil

def get_file_path(output_dir):
    folder_to_compress = output_dir
    compressed_file_name = "flipping"
    compressed_file = shutil.make_archive(compressed_file_name, "zip", folder_to_compress)
    compressed_file_path = os.path.abspath(compressed_file)
    return compressed_file_path

def flipping(directory, output_dir):
    import numpy as np
    from PIL import Image
    import os
    import random
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    image_count = 1
    # Iterate through all the images in the directory
    for filename in os.listdir(directory):
        # Open the image
        with Image.open(os.path.join(directory, filename)) as img:
            img = np.fliplr(img)
            name = "/image_" + str(image_count) + ".jpg"
            cv2.imwrite(output_dir + name, img)
            image_count += 1
            img2 = np.flipud(img)
            name = "/image_" + str(image_count) + ".jpg"
            cv2.imwrite(output_dir + name, img2)
            image_count += 1
    st.write("Random Flipping is done")

def random_rotation(input_dir, output_dir):
    import os
    import random
    from PIL import Image
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the angle range for rotation
    angle_range = (-45, 45)

    # Iterate over the images in the input directory
    for filename in os.listdir(input_dir):
        # Open the image
        with Image.open(os.path.join(input_dir, filename)) as img:
            # Generate a random angle for rotation within the angle range
            angle = random.uniform(*angle_range)

            # Rotate the image
            rotated_img = img.rotate(angle)

            # Save the rotated image to the output directory with a new filename
            new_filename = os.path.splitext(filename)[0] + f'_rotated_{angle:.1f}.jpg'
            rotated_img.save(os.path.join(output_dir, new_filename))
    st.write("Rotation is done")

def cropping(input_dir, output_dir):
    import os
    import random
    from PIL import Image
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Define the crop size
    crop_size = (224, 224)

    # Iterate over the images in the input directory
    for filename in os.listdir(input_dir):
        # Open the image
        with Image.open(os.path.join(input_dir, filename)) as img:
            # Get the width and height of the image
            width, height = img.size

            # Calculate the maximum x and y coordinates for the top-left corner of the crop
            max_x = width - crop_size[0]
            max_y = height - crop_size[1]

            # Generate a random x and y coordinate for the top-left corner of the crop
            x = random.randint(0, max_x)
            y = random.randint(0, max_y)

            # Crop the image
            cropped_img = img.crop((x, y, x+crop_size[0], y+crop_size[1]))

            # Save the cropped image to the output directory with a new filename
            new_filename = os.path.splitext(filename)[0] + f'_cropped_{crop_size[0]}x{crop_size[1]}.jpg'
            cropped_img.save(os.path.join(output_dir, new_filename))
    st.write("cropping is done")


def mixup(folder_path,output_dir):
    import os
    import random
    from PIL import Image

    # Get the list of all images in the folder
    images_list = os.listdir(folder_path)
    serial_no = 0
    for filename in os.listdir(folder_path):
            # Open the image
            with Image.open(os.path.join(folder_path, filename)) as img:
                # Select two random images from the list
                image1_path =  os.path.join(folder_path,filename)
                image2_path = os.path.join(folder_path, random.choice(images_list))

                # Load the images using Pillow
                image1 = Image.open(image1_path)
                image2 = Image.open(image2_path)

                # Resize the images to the same size
                width, height = 256, 256
                image1 = image1.resize((width, height))
                image2 = image2.resize((width, height))

                # Mix the images by taking the average of the pixel values
                mixed_image = Image.blend(image1, image2, 0.5)
                
                # Serial no to give the file a unique name
                serial_no += 1

                # Make a new directory if doesnot exist
                os.makedirs(output_dir, exist_ok=True)

                # Save the mixed image
                if os.path.exists(output_dir):
                    new_filename = str(serial_no) +".jpg"
                    mixed_image.save(os.path.join(output_dir, new_filename))
    st.write("Mix-up is Done!")

def random_erasing(input_dir, output_dir):
    def random_erasing_generator(img, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3.3, v_l=0, v_h=255):
        h, w, _ = img.shape
        s = h * w

        while True:
            se = np.random.uniform(sl, sh) * s
            re = np.random.uniform(r1, r2)
            he = int(np.sqrt(se / re))
            we = int(se / he)

            xe = np.random.randint(0, w)
            ye = np.random.randint(0, h)

            if xe + we <= w and ye + he <= h:
                break
        c = np.random.uniform(v_l, v_h, (he, we, 3))
        img[ye:ye + he, xe:xe + we, :] = c
        return img

    # create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # loop through all images in the input directory
    for filename in os.listdir(input_dir):

            # read the image from the input directory
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # perform random erasing
            erased_img = random_erasing_generator(img)
            # save the erased image to the output directory
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, erased_img)
    
    st.write('Random-Erasing is Done!')

def grayscaling(input_dir, output_dir):

    # create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # loop through all images in the input directory
    for filename in os.listdir(input_dir):
            
            # read the image from the input directory
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # save the grayscale image to the output directory
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, gray_img)
    st.write("Grey-Scaling is Done!")

def noise_injection(input_dir, output_dir):
    import cv2
    import numpy as np
    from PIL import Image
    import os

    def add_salt_and_pepper_noise(image, noise_prob=0.05):
        # Adds salt and pepper noise to the input image.
        noisy_image = image.copy()
        h, w, c = image.shape
        for row in range(h):
            for col in range(w):
                for ch in range(c):
                    # Generate random number between 0 and 1
                    r = np.random.rand()
                    if r < noise_prob / 2:
                        # Add salt noise (white pixel)
                        noisy_image[row, col, ch] = 255
                    elif r < noise_prob:
                        # Add pepper noise (black pixel)
                        noisy_image[row, col, ch] = 0
        return noisy_image

    os.makedirs(output_dir, exist_ok=True)
    serial_no = 0
    
    # Iterate over the images in the input directory
    for filename in os.listdir(input_dir):
        img = cv2.imread(os.path.join(input_dir, filename))
        
        # Generate new images with salt and pepper noise
        noisy_img_sp = add_salt_and_pepper_noise(img)
        filename_sp = "noisy_image_salt_and_pepper_" + str(serial_no) + ".jpg"
        noisy_img_sp = Image.fromarray(np.uint8(noisy_img_sp))
        noisy_img_sp.save(os.path.join(output_dir, filename_sp))
        
        serial_no += 1
    st.write("Noise Injection is Done!")

def colour_transformation(input_dir, output_dir):
    import cv2
    import numpy as np
    from matplotlib import pyplot as plt
    import os
    from PIL import Image

    # create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # loop through all images in the input directory
    for filename in os.listdir(input_dir):
            
            # read the image from the input directory
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # convert the image to grayscale
            gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            YCrCb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)

            # save the grayscale image to the output directory
            out_path = os.path.join(output_dir, filename)
            cv2.imwrite(out_path, gray_img)
            cv2.imwrite(out_path, yuv_img)
            cv2.imwrite(out_path, YCrCb_img)
    st.write("Colour - Transformation is Done!")


def kernel_filtering(input_dir,output_dir):
    import cv2
    import numpy as np
    import os

    # Define kernel
    kernel = np.array([[0, -1, 0],[-1, 5, -1],[0, -1, 0]])

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Loop over all images in input directory
    for filename in os.listdir(input_dir):
        # Load image
        img = cv2.imread(os.path.join(input_dir, filename))

        # Apply kernel filter
        filtered_img = cv2.filter2D(img, -1, kernel)

        # Save filtered image to output directory
        cv2.imwrite(os.path.join(output_dir, filename), filtered_img)
    st.write(" Kernel Filtering is Done!")


# Website 
# -------------------------------------------------------------------------------------------------------
#configure the page
st.set_page_config(page_title = "Dataugmento",page_icon="https://icons.getbootstrap.com/icons/database-fill/")
with st.container():
    
    with st.sidebar:
        # Add title
        selected = option_menu(menu_title = "Steps_to_follow", options = ["upload","Select_Method","Download"],orientation = "vertical", icons= ["file-earmark-arrow-up-fill","box-arrow-in-up-right","file-earmark-arrow-down-fill"])
        
# Upload Page     
if selected == "upload":
    st.subheader("Choose Data")
    st.markdown("""---""")
    uploaded_files = st.file_uploader("Please choose a CSV file/Folder",accept_multiple_files = True)
    path_to_upload = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//Raw_Images"
    if uploaded_files:
        for file in uploaded_files:
            filename = file.name
            with open(os.path.join(path_to_upload, filename), "wb") as f:
                f.write(file.getbuffer())
        st.success("Files have been saved successfully!")

# Methods Page
if selected == "Select_Method": 
    st.subheader("Choose which method to implement")
    st.markdown("""---""")
    b1,b2,b3 = st.columns([1,1,1])
    b4,b5,b6 = st.columns([1,1,1])
    b7,b8,b9 = st.columns([1,1,1])

    with b1:
       if st.button("Flipping"):
            # Set the directory containing the imagescd desk
            input_dir = 'C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//bike'
            output_dir = 'C://Users//Honey//Desktop//Image Datasets//Augmented_Images//random_flipping_result'
            flipping(input_dir, output_dir) 
    
    with b2:    
        if st.button("Cropping"):
            input_dir = 'C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//bike'
            output_dir = 'C://Users//Honey//Desktop//Image Datasets//Augmented_Images//cropping_result'
            cropping(input_dir, output_dir) 

    with b3:
        if st.button("Rotation"):
            # Set the directories
            input_dir = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//bike"
            output_dir = "C://Users//Honey//Desktop//Image Datasets//Augmented_Images//rotation_result"
            random_rotation(input_dir, output_dir)

    with b4:
        if st.button("Mix-up"):
            folder_path = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//dogs"
            output_dir = "C://Users//Honey//Desktop//Image Datasets//Augmented_Images//mixup_result"
            mixup(folder_path,output_dir)

    with b5:
        if st.button("Kernel Filters"):
            input_dir = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//dogs"
            output_dir = "C://Users//Honey//Desktop//Image Datasets//Augmented_Images//kernel_filter_result"
            kernel_filtering(input_dir,output_dir)
        
    with b6:
        if st.button("Random-Erasing"):
            input_dir = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//bike"
            output_dir = "C://Users//Honey//Desktop//Image Datasets//Augmented_Images//random-erasing_result"
            random_erasing(input_dir,output_dir)

    with b7:
        if st.button("Grey-Scaling"):
            input_dir = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//bike"
            output_dir = "C://Users//Honey//Desktop//Image Datasets//Augmented_Images//grey-scaling_result"
            grayscaling(input_dir,output_dir)     
    with b8:
        if st.button("Colour - Transformation"):
            input_dir = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//bike"
            output_dir = "C://Users//Honey//Desktop//Image Datasets//Augmented_Images//colour-trasformation_result"
            colour_transformation(input_dir, output_dir)
    with b9:
        if st.button("Noise - Injection"):
            input_dir = "C://Users//Honey//Desktop//Dataugmento//Image Datasets//data//flowers"
            output_dir = "C://Users//Honey//Desktop//Image Datasets//Augmented_Images//noise-injection_result"
            noise_injection(input_dir, output_dir)

# Download Page
if selected == "Download":
    st.subheader("Output is ready")
    st.markdown("---")

    if st.button('Download Folder'):
        output_dir = 'C://Users//Honey//Desktop//Image Datasets//Augmented_Images'
        file_path = get_file_path(output_dir)
        with open(file_path, 'rb') as f:
            data = f.read()
        st.download_button(label = 'Download', data = data, file_name = 'my_folder.zip', mime = 'application/zip')
hide_st_style = """<style>#MainMenu {visibility: hidden;}footer {visibility: hidden;}header {visibility: hidden;}</style>"""
st.markdown(hide_st_style, unsafe_allow_html=True)