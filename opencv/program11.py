import cv2
import numpy as np


image = np.zeros((500,500,3), dtype="uint8")

#line cv2.line(<image>, startPoint, endPoint, color, thickness)
cv2.line(image, (0,0), (500,500), (0,0,255), 5)


#rectangle cv2.rectangle(<image>, startPoint, endPoint, color, thickness(-1 to fillout))
cv2.rectangle(image, (50,50), (200,200), (0,255,0), 5)
cv2.rectangle(image, (200,200), (300,300), (255,0,0), -1) #filled rectangle


#circle cv2.circle(<image>, point, radius, color, thickness(-1 to fillout))
cv2.circle(image, (400,100), 100, (255,255,0), 5)
cv2.circle(image, (100,400), 50, (255,255,255), -1) #filled rectangle

cv2.putText(image, "Rectangle", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

cv2.imshow("Lines", image)
cv2.waitKey(0)