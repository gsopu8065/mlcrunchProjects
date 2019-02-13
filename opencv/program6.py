import cv2

image = cv2.imread("./bill.png")

for i in [1,0,-1]:
    flipped = cv2.flip(image, i)
    cv2.imshow("Flipped", flipped)
    cv2.waitKey(0)

cv2.destroyAllWindows()