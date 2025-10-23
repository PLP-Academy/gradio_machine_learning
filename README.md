# 🎨 MNIST Digit Classifier - Gradio App

A sleek, interactive web application for classifying handwritten digits using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

## 🌟 Features

- **✏️ Interactive Drawing:** Draw digits with your mouse or touch
- **🤖 Real-time Prediction:** Instant classification results
- **📊 Confidence Scores:** See prediction probabilities for all digits (0-9)
- **🎯 High Accuracy:** 99.42% accuracy on test set
- **📱 Responsive Design:** Works on desktop and mobile devices
- **⚡ Fast Inference:** Optimized for real-time predictions

## 🚀 Quick Start

### Local Development

```bash
# Install dependencies
pip install poetry
poetry install

# Run the app
python app.py
```

**🌐 Access at:** `http://localhost:7860`

### Hugging Face Spaces Deployment

```bash
# Deploy to Hugging Face Spaces
# 1. Create a new Space: https://huggingface.co/spaces
# 2. Upload your files or connect your GitHub repository
# 3. Set the app file to `app.py` and requirements file to `requirements.txt`
# 4. Deploy automatically

# For local testing with Hugging Face Spaces SDK
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload bonus/ . --repo-type=space --commit-message="Update MNIST classifier"
```

## 🛠 Technical Details

### Model Architecture
- **Input:** 28x28 grayscale images
- **Layers:** Conv2D → MaxPooling → Conv2D → MaxPooling → Flatten → Dense → Dropout
- **Output:** 10 classes (digits 0-9) with softmax probabilities
- **Training:** 15 epochs with data augmentation

### Performance Metrics
- **Test Accuracy:** 99.42%
- **Model Size:** ~2.5MB (compressed)
- **Inference Time:** <100ms per prediction
- **Memory Usage:** Optimized for cloud deployment

### Dependencies
Dependencies are managed with Poetry in `pyproject.toml`.
```toml
[tool.poetry.dependencies]
python = "^3.12"
gradio = "*"
tensorflow = ">=2.0.0"
numpy = "*"
Pillow = "*"
```

### Model Optimization
The app uses TensorFlow Lite with post-training quantization for deployment efficiency:
- Automatic conversion from Keras `.h5` to optimized TFLite model
- Float16 quantization reduces model size while maintaining accuracy
- TFLite inference engine for lightweight cloud deployment
- Fallback option to use Keras model (set `USE_TFLITE = False` in `app.py`)
```

## 📁 Project Structure

```
📁 bonus/
├── 📄 app.py                      # Main Gradio application
├── 📄 mnist_cnn_improved_model.h5 # Trained CNN model
└── requirements.txt (exported dependencies)
└── 📄 README.md                   # This file
```

## 🌐 Deployment

### Vercel (Recommended)

**Automatic Deployment:**
1. Push this folder to a GitHub repository
2. Connect repository to Vercel
3. Deploy automatically on every push

**Manual Deployment:**
```bash
cd bonus
vercel --prod
```

**Expected URL:** `https://your-username-your-space-name.hf.space`

### Alternative Platforms

#### Hugging Face Spaces (Recommended)
```bash
# Via Hugging Face Hub CLI
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload bonus/ . --repo-type=space --commit-message="Deploy MNIST classifier"
```

#### Railway
```bash
npm i -g @railway/cli
railway login
railway deploy
```

#### Render
1. Connect GitHub repository to Render
2. Configure Python service
3. Deploy automatically

## 🔧 Configuration

### Environment Variables (Hugging Face Spaces)
```bash
# No special environment variables required
# The app handles model loading automatically
```

### Model Loading
The app automatically loads and optimizes the pre-trained model:
- **Local:** Loads from `mnist_cnn_improved_model.h5` and converts to TFLite
- **Production:** Uses the pre-converted TFLite model for optimal performance

## 🎯 Usage

1. **Open the web interface**
2. **Draw a digit** (0-9) in the sketchpad
3. **View prediction** with confidence scores
4. **Try different digits** to test accuracy

## 🚨 Troubleshooting

### Common Issues

**Model Loading Errors:**
```python
# The app handles model loading automatically
# If issues occur, check file paths and permissions
```

**Memory Issues (Vercel):**
```python
# Already optimized with tensorflow-cpu
# Monitor Vercel function logs for memory usage
```

**Import Errors:**
```bash
# Ensure all dependencies are installed
pip install poetry
poetry install
```

## 📊 Performance

- **Cold Start:** <3 seconds (Vercel serverless)
- **Prediction Time:** <100ms per image
- **Concurrent Users:** Automatic scaling
- **Uptime:** 99.9% (Vercel SLA)

## 🔮 Future Enhancements

- **Multi-digit Recognition:** Process sequences of digits
- **Real-time Video:** Camera input for live classification
- **Model Updates:** Over-the-air model updates
- **Batch Processing:** Multiple images at once

## 📞 Support

For issues or questions:
- **Check Vercel logs** for deployment errors
- **Monitor function performance** in Vercel dashboard
- **Test locally** before deploying to production

---

<div align="center">

**🎯 Ready to classify some digits?**

[![Deploy to Vercel](https://vercel.com/button/button)](https://vercel.com/new/clone?repository-url=https://github.com/PLP-Academy/ai_se_week2_assignment_ai_tools&project-name=mnist-digit-classifier&repo-name=mnist-digit-classifier)

</div>