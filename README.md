# Traffic Detection System

A YOLOv5-based traffic detection system for vehicles and pedestrians, trained on a diverse dataset of traffic camera images from various locations.

## Project Overview

This project implements a computer vision system for detecting and analyzing traffic objects including vehicles and pedestrians. The system is built using YOLOv5 architecture and trained on a comprehensive dataset of traffic camera images.

### Features

- Real-time object detection for vehicles and pedestrians
- Support for multiple object classes
- High accuracy with MAP of 0.89
- Trained on diverse environmental conditions
- Easy to use inference pipeline

## Project Structure

```
traffic-detection-system/
├── data/
│   ├── train/
│   ├── valid/
│   └── test/
├── src/
│   ├── data_processing/
│   ├── model/
│   └── utils/
├── configs/
├── notebooks/
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/AJTHO21/traffic-detection-system.git
   cd traffic-detection-system
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Dataset

The dataset contains traffic camera images with the following characteristics:
- Size: 611MB
- Split into train, validation, and test sets
- Annotated with bounding boxes for vehicles and pedestrians
- Diverse environmental conditions and geographical locations
- Achieved metrics: mAP=0.89, Precision=0.88, Recall=0.89

## Usage

[Usage instructions will be added as the project develops]

## Model Training

[Training instructions will be added as the project develops]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[License information to be added]

## Contact

For questions about the project or dataset, please open an issue on GitHub. 