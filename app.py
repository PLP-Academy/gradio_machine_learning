import gradio as gr
import tensorflow as tf
import numpy as np

# =============================================================================
# 1. Load the Pre-trained MNIST CNN Model
# =============================================================================
# The model was trained and saved in 'practical/tensorflow/mnist_cnn.ipynb'.
# It's a Convolutional Neural Network designed to classify handwritten digits (0-9).

USE_TFLITE = True  # Change to False if you want to use the Keras model instead

try:
    # Load the Keras model
    keras_model = tf.keras.models.load_model('mnist_cnn_improved_model.h5')
    print("✅ MNIST CNN model loaded successfully!")

    # Convert to TensorFlow Lite with post-training quantization
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Use float16 for quantization
    tflite_model = converter.convert()

    # Save the TFLite model
    with open('mnist_cnn_improved_model.h5', 'wb') as f:
        f.write(tflite_model)
    print("✅ Model converted to TFLite with quantization!")

    # Create TFLite interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

except Exception as e:
    print(f"❌ Error loading, converting, or initializing model: {e}")
    print("Please ensure 'mnist_cnn_improved_model.h5' exists and is a valid Keras model.")
    exit()

# =============================================================================
# 2. Define the Prediction Function for Gradio
# =============================================================================
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
        return {str(i): 0.0 for i in range(10)}

    # Extract the composite image from the Sketchpad dict (RGBA format)
    composite_image = image['composite']

    if composite_image is None:
        return {str(i): 0.0 for i in range(10)}

    # Convert to grayscale
    rgb_image = composite_image[:, :, :3]
    gray_image = np.dot(rgb_image, [0.2989, 0.5870, 0.1140])

    # Invert colors: MNIST expects dark background, white digits
    inverted_image = 255.0 - gray_image

    # Resize to 28x28
    image_resized = tf.image.resize(inverted_image[..., np.newaxis], (28, 28)).numpy()
    image_norm = image_resized / 255.0
    image_reshaped = np.expand_dims(image_norm, axis=0).astype(np.float32)

    # Make prediction (use TFLite or Keras)
    if USE_TFLITE:
        interpreter.set_tensor(input_details[0]['index'], image_reshaped)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    else:
        predictions = keras_model.predict(image_reshaped)[0]

    # Threshold check for "not a digit"
    MAX_CONFIDENCE_THRESHOLD = 0.1
    max_conf = np.max(predictions)
    if max_conf < MAX_CONFIDENCE_THRESHOLD:
        return {"Not a digit": 1.0}

    return {str(i): float(predictions[i]) for i in range(10)}

# =============================================================================
# 3. Set up the Gradio Interface
# =============================================================================
input_component = gr.Sketchpad(
    width=280,
    height=280,
    label="Draw a digit (0-9)"
)

output_component = gr.Label(
    num_top_classes=3,
    label="Prediction"
)

iface = gr.Interface(
    fn=classify_digit,
    inputs=input_component,
    outputs=output_component,
    title="MNIST Digit Classifier",
    description="Draw a digit or upload an image, and the CNN model will predict it!",
    live=True,
    allow_flagging='never'
)

# =============================================================================
# 4. Launch the Gradio Application
# =============================================================================
if __name__ == "__main__":
    print("\n🚀 Launching Gradio application locally...")
    print("Please wait for the local URL to appear in your terminal.")
    iface.launch(share=True)

# For serverless environments (e.g., Vercel, HF Spaces)
app = iface.server_app
