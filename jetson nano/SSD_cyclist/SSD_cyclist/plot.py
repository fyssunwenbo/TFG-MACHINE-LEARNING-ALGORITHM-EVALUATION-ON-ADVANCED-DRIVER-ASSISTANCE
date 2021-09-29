import matplotlib
import matplotlib.pyplot as plt

def draw(x1, y1, x2, y2):
    plt.plot(x1, y1, 'r', marker='*', ms=10, label='cyclist')
    plt.plot(x2, y2, 'b', marker='x', ms=10, label='person')
    plt.xlabel("epoch")
    plt.ylabel("mAP")
    plt.title("mAP")
    plt.legend(loc="upper left")
    for xx1, yy1 in zip(x1,y1):
        plt.text(xx1, yy1+1, str(yy1), ha='center', va='bottom', fontsize=20)
    for xx1, yy1 in zip(x2,y2):
        plt.text(xx1, yy1+1, str(yy1), ha='center', va='bottom', fontsize=20)

    plt.savefig("ssd_vgg16.jpg")
    plt.show()


if __name__ == "__main__":
    # y1:cyclist
    # y2:person
    x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]  #横坐标
    #---SSD-vgg16----------------
    map = [0.769, 0.81, 0.798, 0.802, 0.798, 0.806, 0.803, 0.803, 0.811, 0.809, 0.810, 0.810]
    y1 = [0.747,  0.815, 0.801, 0.806, 0.826, 0.821, 0.818, 0.834, 0.855, 0.850, 0.851, 0.851]
    y2 = [0.79, 0.804, 0.795, 0.797, 0.769, 0.791, 0.787, 0.772, 0.768, 0.767, 0.768, 0.769]
    #---------SSD-mobile-lite
    # map=[0.6117,0.6493, 0.6576,0.6693,0.6565, 0.6517, 0.6632, 0.6476,  0.6680, 0.6648, 0.6697, 0.6670]
    # y1=[0.4253,0.4913,0.5040,0.5396,0.4987, 0.4979, 0.5187, 0.4936, 0.5335, 0.5330, 0.5411, 0.5370]
    # y2=[0.7981,0.8074, 0.8111, 0.7990, 0.8143, 0.8054, 0.8078,0.8016, 0.8025,  0.7967,0.7982, 0.7971]
    draw(x,y1,x,y2)