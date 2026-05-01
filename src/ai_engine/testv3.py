import torch
import numpy as np
from tracknetv3 import TrackNetV3, ShuttlecockTrackerV3
import os

def test_model_loading():
    model_path = r"D:\py_projects\badminton_cutter\weights\TrackNet_best.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"--- 正在测试模型加载 ---")
    print(f"检查权重文件是否存在: {os.path.exists(model_path)}")
    
    try:
        # 1. 尝试直接加载权重查看结构
        checkpoint = torch.load(model_path, map_location=device)
        print("权重文件加载成功！")
        
        # 打印权重里的 Key，看看模型是在哪个 Key 下面
        if isinstance(checkpoint, dict):
            print(f"权重包含的 Keys: {checkpoint.keys()}")
        
        # 2. 实例化模型并加载
        # 注意：这里的 TrackNetV3 类必须已经你在 tracknetv3.py 中定义好
        model = TrackNetV3(in_dim=27, out_dim=8).to(device)
        
        # 处理可能的 key 包裹情况
        state_dict = checkpoint.get('model', checkpoint)
        model.load_state_dict(state_dict)
        print("✅ 模型结构与权重完全匹配，加载成功！")
        
        # 3. 模拟一次推理过程
        print(f"--- 正在测试模拟推理 ---")
        
        # 修正：将 12 改为 27，以匹配模型要求的 9帧 RGB 输入 (9 * 3 = 27)
        dummy_input = torch.randn(1, 27, 288, 512).to(device)
        
        model.eval()
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"推理成功！输出维度: {output.shape}")
        
        # 修正：权重文件输出维度为 8 (8通道热力图)[cite: 2]
        if output.shape == (1, 8, 288, 512):
            print("✅ 输出维度符合该权重文件的 TrackNetV3 标准。")
        else:
            print(f"⚠️ 输出维度 {output.shape} 与预期 (1, 8, 288, 512) 不符。")

    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_loading()