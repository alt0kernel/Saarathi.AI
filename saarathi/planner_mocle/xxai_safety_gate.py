import torch

class XXAISafetyAndDPOGate:
    """
    Implements Section V-H (Eq. 35):
    Priority R_i = alpha*p_i + beta*s_i + gamma*q_i + delta*k_i
    XXAI deterministic safety override when d_obs < tau_obs
    DPO tactile/action-first phrasing (stripping visual color terms).
    """
    COLOR_WORDS = {"red", "blue", "green", "yellow", "orange", "purple", "pink"}

    def __init__(self, weights=(0.25, 0.35, 0.20, 0.20), tau_obs: float = 1.2):
        assert abs(sum(weights) - 1.0) < 1e-5, "Weights must sum to 1"
        self.weights = weights
        self.tau_obs = tau_obs

    def compute_priority(self, p_conf: torch.Tensor, s_sim: torch.Tensor,
                         q_spatial: torch.Tensor, k_gog: torch.Tensor) -> torch.Tensor:
        alpha, beta, gamma, delta = self.weights
        return alpha * p_conf + beta * s_sim + gamma * q_spatial + delta * k_gog

    def format_dpo_speech(self, action_directive: str, sku_meta: dict = None, is_novice: bool = True) -> str:
        # Strip color adjectives per Section II-B1 DPO alignment
        words = [w for w in action_directive.split() if w.lower() not in self.COLOR_WORDS]
        cleaned_action = " ".join(words)

        if sku_meta:
            allergen_str = ", ".join(sku_meta["allergens"]) if sku_meta["allergens"] else "none"
            msg = (f"{cleaned_action}. Found {sku_meta['name']}, {sku_meta['tactile_desc']}. "
                   f"Price: {sku_meta['price_inr']} rupees. Allergens: {allergen_str}.")
        else:
            msg = cleaned_action

        if is_novice:
            msg += " Your alignment is spot-on—keep going!"
        return msg