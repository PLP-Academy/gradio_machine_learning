import gradio as gr  # type: ignore
import tensorflow as tf  # type: ignore
import numpy as np

# =============================================================================
# 1. Load the Pre-trained MNIST CNN Model
# =============================================================================
# The model was trained and saved in 'practical/tensorflow/mnist_cnn.ipynb'.
# It's a Convolutional Neural Network designed to classify handwritten digits (0-9).
try:
    model = tf.keras.models.load_model('mnist_cnn_improved_model.h5')
    print("✅ MNIST CNN model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Please ensure 'mnist_cnn_model.h5' exists and is a valid Keras model.")
    # Exit or handle the error appropriately if the model is crucial for the app
    exit()

# =============================================================================
# 2. Define the Prediction Function for Gradio
# =============================================================================
# This function takes a grayscale image as input, preprocesses it,
# and then uses the loaded model to predict the digit.
def classify_digit(image):
    """
    Classifies a handwritten digit image using the pre-trained TensorFlow CNN model.

    Args:
        image (np.ndarray): A 28x28 grayscale image (from Gradio's Image component).

    Returns:
        dict: A dictionary where keys are digit labels (0-9) and values are
              their corresponding prediction probabilities.
    """
    if image is None:
        return {str(i): 0.0 for i in range(10)} # Return zero probabilities if no image

    # Extract the composite image from the Sketchpad dict (RGBA format)
    composite_image = image['composite'] # Shape: (H, W, 4) - RGBA

    # Check if no drawing is done (composite is None)
    if composite_image is None:
        return {str(i): 0.0 for i in range(10)} # Return zero probabilities if no image drawn

    # Convert to grayscale (since MNIST is grayscale)
    # Take RGB channels and convert to gray using luminance formula
    rgb_image = composite_image[:, :, :3]  # (H, W, 3)
    gray_image = np.dot(rgb_image, [0.2989, 0.5870, 0.1140])  # Luminance: 0.3*R + 0.59*G + 0.11*B → (H, W)

    # Invert colors: MNIST expects dark background, white digits
    # (Drawn image has white background, black digits)
    inverted_image = 255.0 - gray_image

    # Resize to 28x28 using TensorFlow and create (H, W, 1)
    image_resized = tf.image.resize(inverted_image[..., np.newaxis], (28, 28)).numpy() # (28, 28, 1)

    # Normalize pixel values to [0, 1]
    image_norm = image_resized / 255.0

    # Reshape for the model: add batch dimension → (1, 28, 28, 1)
    image_reshaped = np.expand_dims(image_norm, axis=0)

    # Make prediction
    predictions = model.predict(image_reshaped)[0] # Get probabilities for the single image

    # Check if it's likely not a digit (max confidence < threshold)
    max_conf = np.max(predictions)
    if max_conf < 0.1:  # Threshold for low confidence
        return {"Not a digit": 1.0}  # Return special output for non-digits

    # Format predictions for Gradio output
    # Create a dictionary mapping digit labels (0-9) to their probabilities
    return {str(i): float(predictions[i]) for i in range(10)}

# =============================================================================
# 3. Set up the Gradio Interface
# =============================================================================
# Gradio provides a simple way to create web interfaces for ML models.
# We define the input and output components, and link them to our prediction function.

# Input component: A sketchpad for drawing digits.
input_component = gr.Sketchpad(
    width=280, # Width for sketching area
    height=280, # Height for sketching area
    label="Draw a digit (0-9)"
)

# Output component: A label to display the predicted digit and confidence.
output_component = gr.Label(
    num_top_classes=3, # Show top 3 most probable classes
    label="Prediction"
)

# Create the Gradio Interface
# The interface connects the input, function, and output.
iface = gr.Interface(
    fn=classify_digit, # The function to run when input changes
    inputs=input_component, # The input component
    outputs=output_component, # The output component
    title="MNIST Digit Classifier", # Title of the Gradio app
    description="Draw a digit or upload an image, and the CNN model will predict it!", # Description
    live=True, # Update predictions live as the user draws/changes input
    allow_flagging='never' # Disable flagging feature
)

# =============================================================================
# 4. Launch the Gradio Application
# =============================================================================
# The 'launch()' method starts the web server for the Gradio app.
# It will typically open in your default browser at a local URL (e.g., http://127.0.0.1:7860).
if __name__ == "__main__":
    print("\n🚀 Launching Gradio application...")
    print("Please wait for the local URL to appear in your terminal.")
    iface.launch(share=False) # Set share=True to get a public link (optional, for sharing)
