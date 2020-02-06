import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def color_dominant(coordinates, image):
    def find_histogram(clt):
        """
        create a histogram with k clusters
        :param: clt
        :return:hist
        """
        numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
        (hist, _) = np.histogram(clt.labels_, bins=numLabels)

        hist = hist.astype("float")
        hist /= hist.sum()

        return hist

    #image = cv2.imread("human.jpg")
    def plot_colors2(hist, centroids):
        bar = np.zeros((50, 300, 3), dtype="uint8")
        startX = 0
        percent_list= []
        for (percent, color) in zip(hist, centroids):
            print(percent, color)
            print("########################")
            percent_list.append(percent)
            # plot the relative percentage of each cluster
            endX = startX + (percent * 300)
            cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                          color.astype("uint8").tolist(), -1)
            startX = endX
        print("percentages are: ", percent_list)
        print("Maximum colour value: ", max(percent_list))
        # return the bar chart
        return bar

    #cv2.imshow("image", img)
        #fromCenter = False
        #r = cv2.selectROI("Image",image, fromCenter)  # returns in existing window

        #imCrop = coordinates[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
    # Display cropped image0
    #cv2.imshow("Image", imCrop)
    img1 = cv2.cvtColor(coordinates, cv2.COLOR_BGR2RGB)

    img1 = img1.reshape((img1.shape[0] * img1.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img1)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)
    '''plt.axis("off")
    plt.imshow(bar)
    plt.show()'''


