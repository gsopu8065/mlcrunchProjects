import cv2

image = cv2.imread('./shapes.jpg')
cv2.imshow("Original Sign", image)

# step1: convert to grayscala
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# step2: blur the image
blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# step3: thresholding
(thresholdValue, thresh) = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
cv2.imshow("Threshold", thresh)

# step4: contours
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if len(contours) == 2 else contours[1]

# step5: find vertices
for c in contours:

    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04 * peri, True)

    shape = ""
    if len(approx) == 3:
        shape = "triangle"

    elif len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)
        diff = abs(w - h)

        if diff < 10:
            shape = "square"
        else:
            shape = "rectangle"

    else:
        shape = "circle"

    # draw contour
    cv2.drawContours(image, [c], -1, (0, 255, 0), 2)

    # put text in the ceneter of the contour
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    cv2.putText(image, shape, (cX - 40, cY), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 4)

cv2.imshow("Shape Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
