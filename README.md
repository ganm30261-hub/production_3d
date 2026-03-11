# production_3d

> **Automatic Reconstruction of Semantic 3D Models from 2D Floor Plans**

`Python` · `PyTorch` · `Docker` · `GitHub Actions` · `Google Cloud Platform` · `Vertex AI Gemini` · `Streamlit` · `MLflow`

---

## 1. Project Overview

`production_3d` is a production-grade AI pipeline that automatically reconstructs semantic 3D models from 2D architectural floor plans. The system implements and extends the method proposed in *"Automatic Reconstruction of Semantic 3D Models from 2D Floor Plans"* (Fraunhofer HHI, MVA 2023).

The pipeline covers the complete workflow from raw floor plan images to 3D models:

- **Data processing** — SVG annotation parsing and multi-format conversion (COCO / YOLO)
- **Wall segmentation** — FPN + ResNet50 with Affinity Loss + Kendall multi-task weighting
- **Door/window detection** — Faster R-CNN bounding box detection
- **Vectorization** — Hough transform + Shrinking algorithm + RDP simplification
- **3D reconstruction** — GLB / IFC model export
- **Experiment tracking** — MLflow metrics and model versioning
- **AI reporting** — Vertex AI Gemini 1.5 Pro + Streamlit interactive dashboard

---

## 2. Project Structure

```
production_3d/
├── b_data/                   # Data processing
│   ├── cubicasa_parser.py    # CubiCasa5k SVG parser
│   ├── coco_converter.py     # COCO format converter
│   ├── yolo_converter.py     # YOLO format converter
│   ├── preprocessing.py      # Crop / sliding window / augmentation
│   ├── pipeline.py           # End-to-end data pipeline
│   └── schemas.py            # Data structure definitions
├── c_models/                 # Model definitions
│   ├── backbone/             # ResNet FPN backbone
│   ├── losses/               # Affinity loss / Kendall loss
│   ├── wall_seg_trainer.py   # Wall segmentation trainer
│   ├── symbol_det_trainer.py # Door/window detection trainer
│   └── yolo_trainer.py       # YOLOv8 enhanced trainer
├── d_vectorization/          # Vectorization
│   ├── hough_transform.py    # Angle detection
│   ├── shrinking_algorithm.py# Rectangle fitting
│   └── wall_extractor.py     # Full vectorization pipeline
├── g_training/               # Training entry points
│   └── main.py               # CLI training script
├── webapp/                   # Web application
│   └── report_app.py         # Streamlit + Gemini dashboard
├── scripts/                  # Utility scripts
│   ├── auto_annotate.py      # Semi-automatic annotation
│   ├── fine_tune.py          # Fine-tuning on company data
│   └── generate_report.py    # Word report generation
├── .github/workflows/        # CI/CD
│   └── ci.yml                # GitHub Actions workflow
├── Dockerfile                # Training container
└── requirements.txt
```

---

## 3. System Architecture

| Component | Technology | Purpose |
|-----------|------------|---------|
| Wall Segmentation | FPN + ResNet50 | Pixel-level wall mask prediction |
| Door/Window Detection | Faster R-CNN | Bounding box localization |
| Instance Segmentation | YOLOv8m-seg | All-class detection |
| Vectorization | Hough + Shrinking + RDP | Mask to polygon conversion |
| 3D Reconstruction | trimesh | GLB / IFC model export |
| Experiment Tracking | MLflow | Metrics and model versioning |
| CI/CD | GitHub Actions | Auto build and push Docker image |
| Cloud Training | GCP Compute Engine | GPU model training |
| Storage | Google Cloud Storage | Dataset and checkpoint management |
| Container Registry | Artifact Registry | Docker image repository |
| AI Reporting | Vertex AI Gemini 1.5 Pro | Plain-language report generation |
| Dashboard | Streamlit + Cloud Run | Interactive reporting interface |

---

## 4. Quick Start

### 4.1 Prerequisites

- Python 3.10+
- CUDA-compatible GPU (recommended)
- Google Cloud SDK installed and authenticated
- Docker installed

### 4.2 Installation

```bash
git clone https://github.com/ganm30261-hub/production_3d.git
cd production_3d
pip install -r requirements.txt
```

### 4.3 Data Preparation

Export training data from CubiCasa5k dataset:

```bash
python -m b_data.pipeline \
  --data_root /path/to/cubicasa5k \
  --target all_paper \
  --splits train val
```

Output structure:
```
output/
├── wall_segmentation/   # images/ + masks/
├── symbol_detection/    # images/ + annotations.json
├── yolo_dataset/        # images/ + labels/ + dataset.yaml
└── coco_annotations/    # coco_annotations_train.json
```

### 4.4 Training (Local)

```bash
# Train wall segmentation + door/window detection
python g_training/main.py \
  --mode train_only \
  --train wall_seg symbol_det \
  --wall_seg_epochs 50 \
  --symbol_det_epochs 15 \
  --gpus auto

# Train all three models
python g_training/main.py \
  --mode train_only \
  --train wall_seg symbol_det yolo \
  --gpus 0,1
```

### 4.5 Training (Google Cloud)

```bash
# Pull latest Docker image built by CI/CD
docker pull us-central1-docker.pkg.dev/PROJECT_ID/floorplan-3d-repo/train:latest

# Mount GCS bucket and run training
gcsfuse YOUR_BUCKET_NAME /mnt/gcs

docker run --gpus all \
  -v /mnt/gcs/datasets:/data_volume \
  -v /mnt/gcs/runs:/data_volume/runs \
  us-central1-docker.pkg.dev/PROJECT_ID/floorplan-3d-repo/train:latest \
  python g_training/main.py \
    --mode train_only \
    --wall_seg_output /data_volume/wall_segmentation \
    --symbol_det_output /data_volume/symbol_detection \
    --yolo_output /data_volume/yolo_dataset \
    --save_dir /data_volume/runs
```

---

## 5. CI/CD Pipeline

Every `git push` to the `main` branch automatically triggers:

| Step | Action | Tool |
|------|--------|------|
| 1 | Code quality check (flake8) | GitHub Actions |
| 2 | Build Docker training image | GitHub Actions + Docker |
| 3 | Push image to Artifact Registry | Google Artifact Registry |
| 4 | GCP VM pulls latest image | Manual / scheduled |
| 5 | Training with GCS data mount | Compute Engine GPU |
| 6 | Checkpoints saved to GCS | Google Cloud Storage |

**Required GitHub Secrets:**

| Secret | Value |
|--------|-------|
| `GCP_SA_KEY` | Google Cloud service account JSON key |
| `GCP_PROJECT_ID` | GCP project ID |

---

## 6. Model Performance

Evaluated on CubiCasa5k validation split:

| Model | Metric | Score | Paper Reference |
|-------|--------|-------|-----------------|
| Wall Segmentation (FPN) | IoU mask | 0.81 | 0.81 |
| Wall Vectorization | IoU vect. | 0.80 | 0.80 |
| Door/Window Detection | mAP@50 | TBD after training | — |
| YOLOv8m-seg | mAP@50-95 | TBD after training | — |

---

## 7. Fine-tuning on Company Data

Pre-trained models can be fine-tuned on company-specific floor plans. With 100 company images the recommended workflow is semi-automatic annotation followed by fine-tuning with a frozen backbone.

### 7.1 Semi-automatic Annotation

```bash
# Step 1: Auto-annotate using pre-trained model
python scripts/auto_annotate.py \
  --image_dir ./company_images \
  --model_path ./runs/wall_segmentation/best.pth \
  --output_dir ./company_annotated

# Step 2: Review and correct annotations in Label Studio
pip install label-studio
label-studio start
```

### 7.2 Fine-tune

```bash
python scripts/fine_tune.py \
  --base_model ./runs/wall_segmentation/best.pth \
  --data_dir ./company_data \
  --output_dir ./runs/fine_tuned \
  --freeze_backbone True \
  --epochs 20 \
  --lr 1e-5
```

### 7.3 Expected Results

| Scenario | Expected IoU |
|----------|-------------|
| CubiCasa model on company data (no fine-tune) | 0.60 – 0.70 |
| After fine-tuning on 100 company images | 0.80+ |

---

## 8. AI Reporting Dashboard

An interactive Streamlit dashboard powered by Vertex AI Gemini 1.5 Pro converts technical training metrics into plain-language management reports.

### 8.1 Run Locally

```bash
# Authenticate with GCP
gcloud auth application-default login

# Launch dashboard
streamlit run webapp/report_app.py
```

### 8.2 Deploy to Cloud Run

```bash
gcloud run deploy floorplan-report \
  --image=us-central1-docker.pkg.dev/PROJECT_ID/floorplan-3d-repo/serve:latest \
  --platform=managed \
  --region=us-central1 \
  --memory=2Gi \
  --allow-unauthenticated
```

**Dashboard features:**
- Real-time training metrics visualization
- Natural language Q&A with Gemini about model performance
- One-click Word report export for management

---

## 9. GCP Infrastructure

| GCP Service | Usage |
|-------------|-------|
| Cloud Storage (GCS) | Training datasets, model checkpoints, prediction outputs |
| Artifact Registry | Docker image repository for training and serving containers |
| Compute Engine (GPU) | Model training with NVIDIA T4 / L4 GPU instances |
| Cloud Run | Serverless deployment of Streamlit reporting dashboard |
| Vertex AI | Gemini 1.5 Pro API for AI-powered report generation |

---

## 10. References

- Barreiro et al. (2023). *Automatic Reconstruction of Semantic 3D Models from 2D Floor Plans*. Fraunhofer HHI, MVA 2023.
- Kalervo et al. (2019). *CubiCasa5k: A Dataset and an Improved Multi-Task Model for Floorplan Image Analysis*. SCIA 2019.
- Lin et al. (2017). *Feature Pyramid Networks for Object Detection*. CVPR 2017.
- Ren et al. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*. NeurIPS 2015.
- Ke et al. (2018). *Adaptive Affinity Fields for Semantic Segmentation*. ECCV 2018.
- Kendall et al. (2018). *Multi-Task Learning Using Uncertainty to Weigh Losses*. CVPR 2018.
