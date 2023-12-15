# Background

Our company develops innovative Artificial Intelligence and Computer Vision solutions that revolutionize industries. Machines that can see: We pack our solutions in small yet intelligent devices that can be easily integrated into your existing data flow. Computer vision for everyone: Our devices can recognize faces, estimate age and gender, classify clothing types and colors, identify everyday objects, and detect motion. Technical consultancy: We help you identify use cases of artificial intelligence and computer vision in your industry. Artificial intelligence is the technology of today, not the future.

FlipScan AI is a new mobile document digitization experience for the blind, for researchers, and for everyone else in need of fully automatic, highly fast, and high-quality document scanning in bulk. It is composed of a mobile app, and all the user needs to do is flip pages; everything is handled by FlipScan AI: it detects page flips from a low-resolution camera preview and takes a high-resolution picture of the document, recognizing its corners and crops it accordingly. It dewarps the cropped document to obtain a bird's-eye view, sharpens the contrast between the text and the background, and finally recognizes the text with formatting kept intact, being further corrected by FlipScan AI's ML-powered redactor.

# Data Description

We collected page-flipping videos from smartphones and labeled them as flipping and not flipping. We clipped the videos as short videos and labeled them as flipping or not flipping. The extracted frames are then saved to disk in sequential order with the following naming structure: VideoID_FrameNumber

# Methodology

- Train a custom convolutional neural network model.
- Apply transfer learning using VGG16, ResNet50, and MobileNet V1.
- Finalize the model to ensure a smaller size.
- Build a web interface demo using Gradio.

# Summary

The finalized model achieved an F1-score of 0.99 with a reduced model size of 45 MB.
