import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification
import warnings

# Suppress HuggingFace warnings for clean output
warnings.filterwarnings("ignore")

def test_classifier():
    print("Loading AI Model Architectures...")
    models = {}
    models['synth_proc'] = AutoImageProcessor.from_pretrained('umm-maybe/AI-image-detector')
    models['synth'] = AutoModelForImageClassification.from_pretrained('umm-maybe/AI-image-detector')
    
    models['df_proc'] = AutoImageProcessor.from_pretrained('dima806/deepfake_vs_real_image_detection')
    models['df'] = AutoModelForImageClassification.from_pretrained('dima806/deepfake_vs_real_image_detection')
    
    DEVICE = "cpu"
    models['synth'] = models['synth'].to(DEVICE)
    models['df'] = models['df'].to(DEVICE)

    # Generate a dummy RGB image tensor exactly like Vision server does
    img = Image.new('RGB', (224, 224), color='red')

    # EXECUTE THE PATCHED LOGIC
    artificial_idx = 0
    for k, v in models['synth'].config.id2label.items():
        if 'artificial' in str(v).lower() or 'fake' in str(v).lower() or 'ai' in str(v).lower():
            artificial_idx = int(k)
            break
            
    fake_idx = 0
    for k, v in models['df'].config.id2label.items():
        if 'fake' in str(v).lower() or 'forgery' in str(v).lower():
            fake_idx = int(k)
            break
            
    print(f"Mapped Tensor Indices => AI/Artificial: {artificial_idx}, Face Fake: {fake_idx}")

    try:
        # Patch code for Synth evaluation
        raw_synth_inputs = models['synth_proc'](images=img, return_tensors="pt")
        synth_inputs = {k: v.to(DEVICE) for k, v in raw_synth_inputs.items()}
        
        with torch.no_grad():
            s_logits = models['synth'](**synth_inputs).logits
            s_probs = torch.nn.functional.softmax(s_logits, dim=-1)
            
        print(f"Synthetic ViT  => Shape: {list(s_logits.shape)}, Output Probability: {s_probs[0][artificial_idx].item():.4f}")

        # Patch code for DF evaluation
        raw_df_inputs = models['df_proc'](images=img, return_tensors="pt")
        df_inputs = {k: v.to(DEVICE) for k, v in raw_df_inputs.items()}
        
        with torch.no_grad():
            d_logits = models['df'](**df_inputs).logits
            d_probs = torch.nn.functional.softmax(d_logits, dim=-1)
            
        print(f"Deepface ViT   => Shape: {list(d_logits.shape)}, Output Probability: {d_probs[0][fake_idx].item():.4f}")
        
        synth_score = max(s_probs[0][artificial_idx].item(), d_probs[0][fake_idx].item())
        print(f"\n✅ TEST PASS: Core logic successfully executed without exceptions.")
        print(f"   Final Synthetic Score returned: {synth_score:.4f}")
        
    except Exception as e:
        import traceback
        print(f"❌ PANIC CRASH: {e}")
        print(traceback.format_exc())

if __name__ == '__main__':
    test_classifier()
