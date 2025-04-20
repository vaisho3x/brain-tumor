import tensorflow as tf
import numpy as np
from PIL import Image
import io

# Global model variable (SageMaker loads it once per container)
model = None

# Called automatically by SageMaker when the endpoint container starts
def model_fn(model_dir):
    """
    Load the trained Keras model (.h5) from the SageMaker model directory.
    """
    global model
    model_path = f"{model_dir}/brain_tumor_model.h5"
    model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded from:", model_path)
    return model

# Called automatically when an inference request is received
def input_fn(request_body, request_content_type):
    """
    Convert raw request input (bytes) into a NumPy array suitable for model input.
    Assumes grayscale image sent as raw bytes.
    """
    if request_content_type == 'application/x-image' or request_content_type == 'application/octet-stream':
        try:
            image = Image.open(io.BytesIO(request_body)).convert('L')  # grayscale
            image = image.resize((28, 28))
            img_array = np.array(image).reshape(1, 28, 28, 1) / 255.0
            return img_array
        except Exception as e:
            raise ValueError(f"❌ Invalid image data: {str(e)}")
    else:
        raise ValueError(f"❌ Unsupported content type: {request_content_type}")

# Run prediction logic using the pre-loaded model and preprocessed input
def predict_fn(input_data, model):
    """
    Perform prediction and return a human-readable result.
    """
    prediction = model.predict(input_data)[0][0]  # Single float output from sigmoid
    label = "Tumor Detected" if prediction > 0.5 else "No Tumor"
    return label

# Return the result in response to client (API/Frontend)
def output_fn(prediction, response_content_type):
    """
    Return prediction in a plain string format.
    """
    if response_content_type == 'application/json':
        return prediction
    else:
        return str(prediction)
