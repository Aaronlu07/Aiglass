import cv2
import numpy as np
import os

# 全局深度学习模型缓存
_de_retinex_model = None
_de_retinex_device = "cpu"

def _load_deep_retinex():
    global _de_retinex_model, _de_retinex_device
    if _de_retinex_model is not None:
        return _de_retinex_model
    
    # 尝试加载基于注意力机制的 PyTorch DE-Retinex 模型
    # 如果没有安装 PyTorch 或没有权重，则回退到传统算法
    try:
        import torch
        from algo.de_retinex import DERetinex
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DERetinex().to(device)
        
        # 尝试加载权重 (如果有的话)
        weight_path = os.path.join(os.path.dirname(__file__), "..", "weights", "de_retinex_best.pt")
        if os.path.exists(weight_path):
            model.load_state_dict(torch.load(weight_path, map_location=device))
            print(f"[Enhance] Loaded Deep Retinex model from {weight_path}")
        else:
            print("[Enhance] Deep Retinex initialized with random weights (Warning: Please train it first)")
            
        model.eval()
        _de_retinex_model = model
        _de_retinex_device = device
        return model
    except Exception as e:
        print(f"[Enhance] Failed to load Deep Retinex: {e}")
        _de_retinex_model = False # 标记为失败，不再尝试
        return False

def single_scale_retinex(img, sigma):
    retinex = np.log10(img + 1.0) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma) + 1.0)
    return retinex

def multi_scale_retinex(img, sigma_list):
    retinex = np.zeros_like(img)
    for sigma in sigma_list:
        retinex += single_scale_retinex(img, sigma)
    retinex = retinex / len(sigma_list)
    return retinex

def color_restoration(img, alpha, beta):
    img_sum = np.sum(img, axis=2, keepdims=True)
    color_restoration = beta * (np.log10(alpha * img + 1.0) - np.log10(img_sum + 1.0))
    return color_restoration

def msrcr(img, sigma_list, G, b, alpha, beta, low_clip, high_clip):
    img = np.float64(img) + 1.0
    img_retinex = multi_scale_retinex(img, sigma_list)
    img_color = color_restoration(img, alpha, beta)
    img_msrcr = G * (img_retinex * img_color + b)

    for i in range(img_msrcr.shape[2]):
        img_msrcr[:, :, i] = (img_msrcr[:, :, i] - np.min(img_msrcr[:, :, i])) / \
                             (np.max(img_msrcr[:, :, i]) - np.min(img_msrcr[:, :, i])) * \
                             255

    img_msrcr = np.uint8(np.clip(img_msrcr, 0, 255))
    return img_msrcr

def enhance_frame(bgr: np.ndarray, gamma: float = 1.2, use_retinex: bool = False):
    # Basic metrics
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2].astype(np.float32) / 255.0
    bright = float(v.mean()) / 255.0
    
    # 1. 低光增强：如果亮度足够高，直接用 Gamma 即可
    enhanced = bgr.copy()
    if use_retinex and bright < 0.4: 
        deep_model = _load_deep_retinex()
        if deep_model:
            # 走 PyTorch DE-Retinex 深度学习增强 (满足毕设要求)
            import torch
            try:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                tensor_img = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0).to(_de_retinex_device) / 255.0
                with torch.no_grad():
                    out_tensor, _ = deep_model(tensor_img)
                out_img = out_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255.0
                out_img = out_img.astype(np.uint8)
                enhanced = cv2.cvtColor(out_img, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[Enhance] Deep Retinex inference failed: {e}")
                deep_model = False # 降级
        
        # 如果模型没跑起来，走传统 MSRCR 增强
        if not deep_model:
            try:
                sigma_list = [15, 80, 250]
                G = 5.0
                b = 25.0
                alpha = 125.0
                beta = 46.0
                low_clip = 0.01
                high_clip = 0.99
                
                enhanced = msrcr(bgr, sigma_list, G, b, alpha, beta, low_clip, high_clip)
            except Exception as e:
                print(f"Retinex failed: {e}, falling back to simple gamma")
                invGamma = 1.0 / max(gamma, 0.1)
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                enhanced = cv2.LUT(bgr, table)
    else:
        # Simple Gamma Correction for Edge Mode
        invGamma = 1.0 / max(gamma, 0.1)
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
        enhanced = cv2.LUT(bgr, table)
        
        # CLAHE
        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    # Calculate sharpness
    gray = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F)
    sharp = float(lap.var())
    
    return {"image": enhanced, "value": bright, "gamma": float(gamma), "sharpness": sharp}
