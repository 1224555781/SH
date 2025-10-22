from ultralytics import YOLO

model = YOLO('best.pt')
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model.export(format='onnx', device=device)

print(f"Successfully exported 'best.pt' to 'best.onnx' using {device}   . You can now use this file in the main script.")