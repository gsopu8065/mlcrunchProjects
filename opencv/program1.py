#<script src="https://cdn.rawgit.com/google/code-prettify/master/loader/run_prettify.js"></script>
#<link href="https://jmblog.github.io/color-themes-for-google-code-prettify/themes/atelier-forest-light.min.css" rel="stylesheet" type="text/css"></link>
#<pre class="prettyprint linenums"><code class="language-py">
import cv2

image = cv2.imread('./bill.png', cv2.IMREAD_GRAYSCALE)
print("width: {0} pixels".format(image.shape[1]))
print("height: {0} pixels".format(image.shape[0]))
print("Image shape = {0}".format(image.shape))
print("First pixel value = {0}".format(image[0][0]))

image2 = cv2.imread('./bill.png')
print("width: {0} pixels".format(image2.shape[1]))
print("height: {0} pixels".format(image2.shape[0]))
print("Image shape = {0}".format(image2.shape))
print("First pixel value = {0}".format(image2[0][0]))

#</code></pre>

cv2.imshow('image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()