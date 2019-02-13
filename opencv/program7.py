import cv2

image = cv2.imread("./bill.png")
h,w = image.shape[:2]
print("Before image width = {0}, height = {1}".format(w,h))
cv2.imshow("Before Resize", image)

resizedImage = cv2.resize(image, (500, h))
rh,rw = resizedImage.shape[:2]
print("After image width = {0}, height = {1}".format(rw,rh))
cv2.imshow("After Resize", resizedImage)


#increase width and height by 40%
newWidth = int(w * 140/100)
newHeight = int(h * 140/100)
resizedImage = cv2.resize(image, (newWidth, newHeight))
rh,rw = resizedImage.shape[:2]
print("After image size increase 40% = {0}, height = {1}".format(rw,rh))
cv2.imshow("After image size increase 40%", resizedImage)

#decrease width and height by 60%
newWidth = int(w * 40/100)
newHeight = int(h * 40/100)
resizedImage = cv2.resize(image, (newWidth, newHeight))
rh,rw = resizedImage.shape[:2]
print("After image size decrease 60% = {0}, height = {1}".format(rw,rh))
cv2.imshow("After image size decrease 60%", resizedImage)

cv2.waitKey(0)
cv2.destroyAllWindows()