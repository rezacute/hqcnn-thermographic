"""
FastAPI REST API for Thermal Image Classification
==================================================
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict
import numpy as np
from PIL import Image
import io
import torch

from thermo_classifier.models.classifier import ThermalClassifier
from thermo_classifier.data.transforms import ThermalTransforms


app = FastAPI(
    title="Thermo Classifier API",
    description="REST API for thermal image classification using quantum-enhanced CNNs",
    version="1.0.0"
)

# Global state
app.state.model_path = None
app.state.domain = "medical"
app.state.classifier = None
app.state.transform = None


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    domain: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    domain: str
    device: str


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    if app.state.model_path:
        try:
            app.state.classifier = ThermalClassifier(
                domain=app.state.domain,
                model_type='hq-cnn',
                use_quantum=True,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                checkpoint=app.state.model_path
            )
            app.state.transform = ThermalTransforms.get_inference_transforms()
        except Exception as e:
            print(f"Error loading model: {e}")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Thermo Classifier API",
        "version": "1.0.0",
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if app.state.classifier else "model_not_loaded",
        model_loaded=app.state.classifier is not None,
        domain=app.state.domain,
        device=app.state.classifier.device if app.state.classifier else "none"
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), domain: Optional[str] = Form("medical")):
    """Predict thermal image class
    
    Upload a thermal image and get classification results.
    
    Args:
        file: Image file (PNG, JPG)
        domain: Classification domain ('medical' or 'industrial')
    
    Returns:
        Prediction results with class, confidence, and probabilities
    """
    if not app.state.classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Read and preprocess image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L')
    image_np = np.array(image)
    
    # Transform
    transformed = app.state.transform(image=image_np)
    image_tensor = transformed['image'].unsqueeze(0)
    
    # Predict
    result = app.state.classifier.predict(image_tensor.squeeze(0))
    
    return PredictionResponse(
        predicted_class=result['predicted_class'],
        confidence=result['confidence'],
        probabilities=result['probabilities'],
        domain=app.state.domain
    )


@app.post("/predict_batch", response_model=List[PredictionResponse])
async def predict_batch(files: List[UploadFile] = File(...)):
    """Predict multiple thermal images
    
    Upload multiple thermal images and get classification results.
    """
    if not app.state.classifier:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L')
        image_np = np.array(image)
        
        transformed = app.state.transform(image=image_np)
        image_tensor = transformed['image'].unsqueeze(0)
        
        result = app.state.classifier.predict(image_tensor.squeeze(0))
        
        results.append(PredictionResponse(
            predicted_class=result['predicted_class'],
            confidence=result['confidence'],
            probabilities=result['probabilities'],
            domain=app.state.domain
        ))
    
    return results


@app.get("/classes")
async def get_classes(domain: Optional[str] = None):
    """Get available classes for domain"""
    domain = domain or app.state.domain
    classes = ThermalClassifier.DOMAIN_CLASSES.get(domain)
    
    if not classes:
        raise HTTPException(status_code=400, detail=f"Unknown domain: {domain}")
    
    return {
        "domain": domain,
        "classes": classes
    }


@app.get("/domains")
async def get_domains():
    """Get available domains"""
    return {
        "domains": list(ThermalClassifier.DOMAIN_CLASSES.keys())
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
