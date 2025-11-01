import torch
import torch.nn.functional as F
import joblib
from src.model_def import NNClassifierModel

def load_model_and_encoder():
    le = joblib.load("model/label_encoder.joblib")
    model = NNClassifierModel()
    model.load_state_dict(torch.load("model/khmer_char_model.pth", map_location="cpu"))
    model.eval()
    return model, le

def predict(model, le, x):
    with torch.no_grad():
        outputs = model(x)
        probs = F.softmax(outputs, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = probs[0][pred_idx].item()
    return pred_label, confidence, probs
