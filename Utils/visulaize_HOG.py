import matplotlib.pyplot as plt

from skimage.feature import hog
from skimage import data, exposure

from DataLoader import DataLoader

file_name = "FDDB-fold-01-ellipseList.txt"

Data = DataLoader()

img_list = list(Data.generate_clipped_faces(file_name))
hog_list = []
for i in range(8):
    image = img_list[i]

    fd, hog_image = hog(image, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(2, 2),
                    visualize=True, multichannel=True)

    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

    # ax1.axis('off')
    # ax1.imshow(image, cmap=plt.cm.gray)
    # ax1.set_title('Input image')

    # Rescale histogram for better display
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    hog_list.append(hog_image_rescaled)

    # ax2.axis('off')
    # ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
    # ax2.set_title('Histogram of Oriented Gradients')
    # plt.show()

plt.figure() #设置窗口大小
# plt.suptitle('') # 图片名称
for i in range(8):
    plt.subplot(2,8,i+1), plt.title('Input')
    plt.imshow(img_list[i]), plt.axis('off')
for i in range(8):
    plt.subplot(2,8,8+i+1), plt.title('HOG')
    plt.imshow(hog_list[i]), plt.axis('off')


plt.show()