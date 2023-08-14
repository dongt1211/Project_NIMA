# from PIL import Image
# import os

# # Define the input and output directories
# input_dir = '/home/bkcs/NIMA/images/images/'
# output_dir = '/home/bkcs/NIMA/images/images2/'

# # Loop through each image in the input directory
# for filename in os.listdir(input_dir):
#     # Open the image file using Pillow
#     img = Image.open(os.path.join(input_dir, filename))

#     # Check if the image has an extraneous bytes error
#     try:
#         img.load()
#     except OSError as e:
#         if "extraneous" in str(e):
#             print(f"Image {filename} has an extraneous bytes error")

#             # Remove the extraneous bytes from the image data
#             img_data = list(img.tobytes())
#             for i in range(len(img_data)-1, 0, -1):
#                 if img_data[i] == 255 and img_data[i-1] == 255:
#                     del img_data[i:]
#                     break

#             # Save the image to the output directory
#             img = Image.frombytes(mode=img.mode, size=img.size, data=bytes(img_data))
#             img.save(os.path.join(output_dir, filename))
#             print(f"Fixed {filename} and saved to {output_dir}")
#         else:
#             print(f"Error processing {filename}: {str(e)}")
#     else:
#         # Save the image to the output directory
#         img.save(os.path.join(output_dir, filename))
#         print(f"Processed {filename} and saved to {output_dir}")
import tensorflow as tf
train_dataset = tf.data.TFRecordDataset(['home/bkcs/NIMA/neural-image-assessment-master/weights/nasnet_large_val.tfrecord'])
print(train_dataset)