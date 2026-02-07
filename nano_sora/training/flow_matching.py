import torch

class RectifiedFlow:
    """
    Implements Rectified Flow (Flow Matching) logic.
    Target velocity v = x1 - x0.
    """
    def __init__(self, model):
        self.model = model

    def get_train_tuple(self, x1):
        """
        Samples t ~ U[0, 1] and creates noisy sample xt.
        x0 is Gaussian noise, x1 is data.
        xt = t * x1 + (1 - t) * x0
        Target velocity is (x1 - x0).
        """
        device = x1.device
        n = x1.shape[0]
        t = torch.rand(n, device=device)
        
        x0 = torch.randn_like(x1)
        
        # Reshape t for broadcasting: (batch, 1, 1, 1)
        t_broadcast = t.view(-1, 1, 1, 1)
        
        # xt = (1 - t) * x0 + t * x1
        xt = (1.0 - t_broadcast) * x0 + t_broadcast * x1
        
        # The target velocity is (x1 - x0)
        target = x1 - x0
        
        return xt, t, target

    @torch.no_grad()
    def sample_euler(self, x0, y, steps=50):
        """
        Simple Euler ODE solver for inference.
        dx = v(x, t) dt
        """
        device = x0.device
        dt = 1.0 / steps
        xt = x0
        
        for i in range(steps):
            t = torch.full((x0.shape[0],), i / steps, device=device)
            v_pred = self.model(xt, t, y)
            xt = xt + v_pred * dt
            
        return xt
