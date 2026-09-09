"""
PCW Lesson 1: Introduction to Machine Learning
Complete working solution for Katia Gwaneza Nkurunziza
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris, fetch_openml

print("=" * 80)
print("CORE SECTION: 3.1 Code Practice")
print("=" * 80)

# ============================================================================
# SCRIPT 1: Load, clean, and visualize the Iris dataset
# ============================================================================

print("\n" + "=" * 80)
print("SCRIPT 1: IRIS DATASET")
print("=" * 80)

# STEP 1: LOAD THE DATA
# An iris is a type of flowering plant. This dataset contains measurements
# taken from 150 flowers belonging to three different iris species.
# Load directly as DataFrame using as_frame=True parameter
iris = load_iris(as_frame=True)
flowers = iris.frame.copy()

# Convert numeric target codes to readable species names
flowers['species'] = flowers['target'].map(dict(enumerate(iris.target_names)))

# Display the first five rows so we can inspect the loaded data.
print("\nOriginal Iris data:")
print(flowers.head())

# Display the number of flowers before filtering.
print(f"\nNumber of flowers before filtering: {len(flowers)}")

# STEP 2: CLEAN/FILTER THE DATA
# A sepal is one of the leaf-like structures surrounding the petals of a flower.
# Sepal length is measured in centimeters.
# Keep flowers whose sepal length is 5 cm or less (drop those > 5 cm).
filtered_flowers = flowers.loc[flowers['sepal length (cm)'] <= 5].copy()

print(f"\nNumber of flowers before filtering: {len(flowers)}")
print(f"Number of flowers after filtering: {len(filtered_flowers)}")
print("\nFiltered Iris data:")
print(filtered_flowers.head())

# STEP 3: VISUALIZE THE DATA
# Assign a different color to each iris species.
colors = {
    'setosa': 'tab:blue',
    'versicolor': 'tab:orange',
    'virginica': 'tab:green'
}

# Create one group of points for each species.
for species, group in filtered_flowers.groupby('species'):
    # Each point represents one flower.
    # Horizontal position: sepal length | Vertical position: petal length
    # Color represents the iris species.
    plt.scatter(
        group['sepal length (cm)'],
        group['petal length (cm)'],
        label=species,
        color=colors[species],
        alpha=0.8
    )

# Add labels explaining what the plot represents.
plt.title("Petal Length vs. Sepal Length")
plt.xlabel("Sepal length (centimeters)")
plt.ylabel("Petal length (centimeters)")

# Add a legend connecting each color to an iris species.
plt.legend(title="Iris species")

# Adjust spacing and display the finished plot.
plt.tight_layout()
plt.show()

# ============================================================================
# SCRIPT 2: Load, clean, and visualize the MNIST handwritten digits dataset
# ============================================================================

print("\n" + "=" * 80)
print("SCRIPT 2: MNIST DATASET")
print("=" * 80)

# STEP 1: LOAD THE MNIST DATASET
# MNIST is a collection of 70,000 images of handwritten digits.
# Each image shows one digit from 0 through 9.
# Every image is 28 pixels wide and 28 pixels tall.
# Therefore, every image contains: 28 × 28 = 784 pixels
# Instead of initially storing each image as a square, MNIST stores each image
# as one row containing 784 pixel values.

print("\nLoading MNIST dataset... (this may take a minute on first run)")
try:
    mnist = fetch_openml(
        "mnist_784",
        version=1,
        as_frame=True,
        parser='auto'
    )
    print("MNIST dataset loaded successfully!")
except Exception as e:
    print(f"Error loading MNIST: {e}")
    print("Using alternative: loading smaller MNIST subset from sklearn")
    from sklearn.datasets import load_digits
    mnist_data = load_digits()
    # Convert to format similar to openml
    class MNISTProxy:
        def __init__(self, data, target):
            self.data = pd.DataFrame(data)
            self.target = target
    mnist = MNISTProxy(mnist_data.data, mnist_data.target)

# STEP 2: PUT THE PIXELS INTO A PANDAS DATAFRAME
# mnist.data contains the pixel values.
digits = pd.DataFrame(mnist.data).copy()

# Display the dimensions of the pixel data.
print(f"\nShape of the pixel data: {digits.shape}")
print(f"Number of images: {digits.shape[0]}")
print(f"Number of pixels per image: {digits.shape[1]}")

# STEP 3: ADD THE CORRECT DIGIT LABEL TO EACH IMAGE
# mnist.target contains the correct answer for each image.
digits["label"] = pd.Series(mnist.target).astype(int)

# Display the first five rows.
print("\nFirst five rows:")
print(digits.head())

# Display how many examples there are of every digit before filtering.
print("\nNumber of images for each digit (before filtering):")
print(digits["label"].value_counts().sort_index())

# STEP 4: KEEP ONLY THE DIGITS 3 AND 8
# The isin([3, 8]) expression creates a True/False value for every row:
# True = the image is labeled 3 or 8
# False = the image is some other digit
digits_3_and_8 = digits.loc[
    digits["label"].isin([3, 8])
].copy()

# Reset the row numbers after removing the other digits.
digits_3_and_8 = digits_3_and_8.reset_index(drop=True)

# Display the number of images remaining.
print(f"\nNumber of images remaining after keeping only 3s and 8s: {len(digits_3_and_8)}")

# Check how many 3s and 8s remain.
print("\nImages remaining for each label:")
print(digits_3_and_8["label"].value_counts().sort_index())

# STEP 5: CHOOSE EXAMPLES TO DISPLAY
# Select six examples of handwritten 3s.
examples_of_3 = digits_3_and_8[
    digits_3_and_8["label"] == 3
].sample(
    n=min(6, len(digits_3_and_8[digits_3_and_8["label"] == 3])),
    random_state=42
)

# Select six examples of handwritten 8s.
examples_of_8 = digits_3_and_8[
    digits_3_and_8["label"] == 8
].sample(
    n=min(6, len(digits_3_and_8[digits_3_and_8["label"] == 8])),
    random_state=42
)

# Combine the selected examples into one DataFrame.
examples = pd.concat(
    [examples_of_3, examples_of_8],
    ignore_index=True
)

# STEP 6: CREATE A GRID FOR THE IMAGES
# Create a plotting area containing three rows and four columns.
# This gives us 12 smaller plotting areas called axes: 3 rows × 4 columns = 12 images
figure, axes = plt.subplots(
    nrows=3,
    ncols=4,
    figsize=(8, 7)
)

# STEP 7: TURN EACH ROW BACK INTO AN IMAGE
# axes.flat treats the 3-by-4 collection of plotting areas as one simple list.
# iterrows() gives us one selected DataFrame row at a time.
# zip() pairs each handwritten digit with one plotting area.
for axis, (_, row) in zip(axes.flat, examples.iterrows()):
    # Save the image's correct label before working with its pixels.
    correct_label = int(row["label"])

    # Remove the label because it is not part of the image.
    pixel_values = row.drop(labels="label")

    # Convert the 784 pixel values into a numeric NumPy array.
    pixel_values = pixel_values.to_numpy(dtype=float)

    # The image was stored as one long row of 784 values.
    # Reshape those values back into the original 28-by-28 square.
    image = pixel_values.reshape(28, 28)

    # imshow() converts the numeric pixel values into a visible image.
    # cmap="gray" displays low values as dark pixels and high values as light pixels.
    # interpolation="nearest" keeps the individual pixels sharp instead of smoothing.
    axis.imshow(
        image,
        cmap="gray",
        interpolation="nearest"
    )

    # Show the correct digit above the image.
    axis.set_title(f"Label: {correct_label}")

    # Hide the coordinate lines and tick marks because this is an image, not a graph.
    axis.axis("off")

# STEP 8: FINISH AND DISPLAY THE PLOT
# Add a title for the entire collection of images.
figure.suptitle(
    "Different Handwritten Examples of 3 and 8",
    fontsize=16
)

# Adjust the spacing so the images and titles do not overlap.
plt.tight_layout()

# Display the completed grid.
plt.show()

print("\n" + "=" * 80)
print("CORE SECTION COMPLETE")
print("=" * 80)
