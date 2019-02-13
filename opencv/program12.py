
import cv2

image = cv2.imread('./bill.png')


def rgb():
    cv2.imshow("RGB", image)
    (B, G, R) = cv2.split(image)
    cv2.imshow("Red", R)
    cv2.imshow("Green", G)
    cv2.imshow("Blue", B)

    merged = cv2.merge([B, G, R])
    cv2.imshow("Merged", merged)

    print(B.shape)
    print(G.shape)
    print(R.shape)
    print(merged.shape)
    print(B[0][0])
    print(G[0][0])
    print(R[0][0])
    print(merged[0][0])

def hsv():
    # convert the image to the HSV color space and show it
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    cv2.imshow("HSV", hsv)
    (H, S, V) = cv2.split(image)
    cv2.imshow("Hue", H)
    cv2.imshow("Saturation", S)
    cv2.imshow("Value", V)

    merged = cv2.merge([H, S, V])
    cv2.imshow("HSV Merged", merged)

    print(H.shape)
    print(S.shape)
    print(V.shape)
    print(merged.shape)
    print(H[0][0])
    print(S[0][0])
    print(V[0][0])
    print(merged[0][0])

def lab():
    # convert the image to the L*a*b* color space and show it
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cv2.imshow("LAB", lab)
    (L, A, B) = cv2.split(image)
    cv2.imshow("L", L)
    cv2.imshow("A", A)
    cv2.imshow("B", B)

    merged = cv2.merge([L, A, B])
    cv2.imshow("LAB Merged", merged)

    print(L.shape)
    print(A.shape)
    print(B.shape)
    print(merged.shape)
    print(L[0][0])
    print(A[0][0])
    print(B[0][0])
    print(merged[0][0])

def gray():
    # show the original and grayscale versions of the image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Gray scale", gray)

cv2.waitKey(0)