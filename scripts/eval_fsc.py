from torch.utils.data import DataLoader
import os
from yolo_count.models.yolocount import build_yolocount_model_base
from yolo_count.utils.validation import evaluate_on_fsc
from yolo_count.utils.dataload import FSCData
from yolo_count.utils.fn import auto_load


os.environ["TOKENIZERS_PARALLELISM"] = "false"

ckpt_path = "checkpoints/yolocnt_fsc147_epoch300.pth"

model = build_yolocount_model_base()
auto_load(model, ckpt_path)
model.eval().to("cuda")

large_threshold = 130
confidence_threshold = 0.0

val_dataloader = DataLoader(
    FSCData(root="data/FSC", split="val", flip=False),
    batch_size=1,
    num_workers=4,
    shuffle=False,
)
test_dataloader = DataLoader(
    FSCData(root="data/FSC", split="test", flip=False),
    batch_size=1,
    num_workers=4,
    shuffle=False,
)

MAE_test, RMSE_test, Bias_test = evaluate_on_fsc(
    model,
    test_dataloader,
    large_threshold=large_threshold,
    confidence_threshold=confidence_threshold,
)
print(f"Test - MAE: {MAE_test:.4f}, RMSE: {RMSE_test:.4f}, Bias: {Bias_test:.4f}")

MAE_val, RMSE_val, Bias_val = evaluate_on_fsc(
    model,
    val_dataloader,
    large_threshold=large_threshold,
    confidence_threshold=confidence_threshold,
)
print(f"Validation - MAE: {MAE_val:.4f}, RMSE: {RMSE_val:.4f}, Bias: {Bias_val:.4f}")
