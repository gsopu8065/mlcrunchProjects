import cv2

image1 = cv2.imread("./bill.png")
image2 = cv2.imread("./logo.png")
h, w = image1.shape[:2]

image2 = cv2.resize(image2, (w,h))
dst = cv2.addWeighted(image1,0.7,image2,0.3,0)

cv2.imshow('dst',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()