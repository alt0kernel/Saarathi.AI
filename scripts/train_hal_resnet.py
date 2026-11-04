import torch
from saarathi.retrieval_sku.resnet_hal import ResNet34HAL

def train_hal_demo(steps: int = 5):
    model = ResNet34HAL(embedding_dim=512, num_fine_classes=20, num_macro_classes=5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    model.train()
    for step in range(1, steps + 1):
        dummy_crops = torch.randn(8, 3, 224, 224)
        fine_labels = torch.randint(0, 20, (8,))
        macro_labels = fine_labels // 4

        optimizer.zero_grad()
        z = model(dummy_crops)
        loss = model.compute_hal_loss(z, fine_labels, macro_labels)
        loss.backward()
        optimizer.step()
        print(f"[Step {step}/{steps}] HAL + CosFace Loss: {loss.item():.4f}")

if __name__ == "__main__":
    train_hal_demo()