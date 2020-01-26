import cv2
import numpy as np


def bitWise():
    rectangle = cv2.imread("./rectangle.jpeg")
    circle = cv2.imread("./circle.jpeg")

    bitwiseAnd = cv2.bitwise_and(rectangle, circle)
    cv2.imshow("AND", bitwiseAnd)


    bitwiseOr = cv2.bitwise_or(rectangle, circle)
    cv2.imshow("OR", bitwiseOr)


    bitwiseXor = cv2.bitwise_xor(rectangle, circle)
    cv2.imshow("XOR", bitwiseXor)


    bitwiseNot = cv2.bitwise_not(rectangle)
    cv2.imshow("NOT", bitwiseNot)

def masking():
    image = cv2.imread("./bill.png")
    blackImage = np.zeros(image.shape[:2], dtype='uint8')
    circle = cv2.circle(blackImage, (190,140), 100, 255, -1)
    cv2.imshow("circle", circle)
    cv2.imshow("bill Face", image)
    maskedImage = cv2.bitwise_and(image, image, mask=circle)
    cv2.imshow("Mask Applied to Image", maskedImage)

print(cv2.bitwise_and(cv2.UMat(np.array([10, 255, 50])), cv2.UMat(np.array([25, 285, 13]))).get())
#masking()
cv2.waitKey(0)
cv2.destroyAllWindows()


#Reference: https://www.jianshu.com/p/003e9451cdc4