import torch
from data.audio_corpus_loader import AudioSpectrogramExtractor
from saarathi.audio_apr.whisper_lora import LoRAWhisperASR
from saarathi.audio_apr.llm_rag_refiner import StoreRAGRefiner
from saarathi.goal_gog.gog_parser import GoalOrientedGraphPlanner
from saarathi.navigation.dijkstra_planner import StoreNavigationGraph
from saarathi.navigation.webarnav_pnp import HybridStoreLocalizer
from saarathi.navigation.obstacle_fusion import MultiSensorObstacleFusion
from saarathi.vision_vpd.yolo_dual_label import VPDDetector
from saarathi.retrieval_sku.resnet_hal import ResNet34HAL
from saarathi.retrieval_sku.caqe_expansion import caqe_refine_embeddings, retrieve_sku_from_catalogue
from saarathi.planner_mocle.dual_encoder import DualBranchVisualEncoder
from saarathi.planner_mocle.patch_info_mining import PatchInformationMining
from saarathi.planner_mocle.mocle_router import MoCLERouter
from saarathi.planner_mocle.xxai_safety_gate import XXAISafetyAndDPOGate

def run_closed_loop_simulation():
    # Initialize all Saarathi AI subsystems
    spec_extractor = AudioSpectrogramExtractor()
    asr = LoRAWhisperASR()
    rag_refiner = StoreRAGRefiner("configs/store_catalog_db.json")
    gog_planner = GoalOrientedGraphPlanner()
    store_nav = StoreNavigationGraph()
    localizer = HybridStoreLocalizer()
    obs_fusion = MultiSensorObstacleFusion(tau_obs=1.2)
    vpd = VPDDetector()
    retriever = ResNet34HAL(embedding_dim=512).eval()
    dual_enc = DualBranchVisualEncoder(d_model=512).eval()
    pim = PatchInformationMining(d_model=512).eval()
    mocle = MoCLERouter(d_model=512).eval()
    safety_gate = XXAISafetyAndDPOGate(tau_obs=1.2)

    # Step 1 (Alg. 1): Identify Requirement
    raw_wave = torch.randn(1, 16000 * 2)
    s_mel = spec_extractor(raw_wave)
    raw_text = asr.transcribe(s_mel, simulated_text="find amul unsalted butter")
    refined_q, context_skus, t_tokens = rag_refiner.refine(raw_text, user_dietary_pref="salted")
    target_sku = context_skus[0]
    G_tuple = gog_planner.parse_request_to_tuple(refined_q, "Entrance", target_sku)
    print(f"1. Refined Request: '{refined_q}' -> GoG Tuple: {G_tuple[0]} | Dest: {G_tuple[2]}")
    print(f"   Procedural Chain: {G_tuple[3]}")

    # Simulate 3 closed-loop time steps: [Hazard Preemption -> Navigation -> Shelf Reach & CAQE]
    sim_steps = [
        {"rgbd_m": 0.9, "tof_m": 0.85, "dist_shelf": 6.0, "node": "Entrance"},
        {"rgbd_m": 3.5, "tof_m": 3.20, "dist_shelf": 3.5, "node": "Aisle_1"},
        {"rgbd_m": 2.8, "tof_m": 2.90, "dist_shelf": 1.1, "node": "Aisle_2"},
    ]

    for idx, step in enumerate(sim_steps, 1):
        localizer.update_pdr(step_length=0.7, delta_heading_rad=0.02)
        frame = torch.rand(1, 3, 480, 640)
        i_enh, i_ann, detections, _ = vpd(frame)
        d_obs, is_hazard = obs_fusion.estimate_min_obstacle_distance(step["rgbd_m"], step["tof_m"])

        # Branch 1: XXAI Hard Safety Rule (Alg. 1 Line 5)
        if is_hazard:
            store_nav.mark_edge_blocked("Entrance", "Aisle_1")
            route, _ = store_nav.shortest_route(step["node"], G_tuple[2])
            speech = safety_gate.format_dpo_speech(
                f"Stop immediately. Obstacle at {d_obs:.2f} meters. Rerouting via {route[1]}"
            )
            print(f"[Step {idx} | XXAI Override] {speech}")

        # Branch 2: MoCLE Navigation Cluster S2 (Alg. 1 Line 7)
        elif step["dist_shelf"] > 1.5:
            with torch.no_grad():
                y_low, y_high = dual_enc(i_ann)
                t_v = pim(y_low, y_high)
                _, _, cluster_id = mocle(t_v, t_tokens)
            route, cost = store_nav.shortest_route(step["node"], G_tuple[2])
            speech = safety_gate.format_dpo_speech(
                f"Turn right toward {G_tuple[2]}, {step['dist_shelf']:.1f} meters ahead"
            )
            print(f"[Step {idx} | MoCLE Cluster S{cluster_id+1} Nav] {speech}")

        # Branch 3: Target Shelf Reached -> Run Algorithm 2 (HAL + CAQE)
        else:
            with torch.no_grad():
                crops = torch.randn(len(detections), 3, 224, 224)
                boxes = torch.tensor([d["box"] for d in detections], dtype=torch.float32)
                z = retriever(crops)
                z_tilde = caqe_refine_embeddings(z, boxes)

                # Build reference catalogue embeddings where index 0 matches target_sku
                cat_z = torch.randn(3, 512)
                cat_z[0] = z_tilde[0]
                cat_z = torch.nn.functional.normalize(cat_z, p=2, dim=-1)

                p_conf = torch.tensor([d["conf"] for d in detections])
                s_sim = torch.matmul(z_tilde, cat_z.T).max(dim=-1).values
                q_spat = torch.tensor([0.9, 0.7, 0.5])
                k_gog = torch.ones_like(p_conf)
                priorities = safety_gate.compute_priority(p_conf, s_sim, q_spat, k_gog)

                j_star, sim_score, verified = retrieve_sku_from_catalogue(z_tilde, cat_z, priorities, tau_sim=0.75)
                if verified:
                    speech = safety_gate.format_dpo_speech(
                        "Reach forward 40 centimeters at chest level", sku_meta=target_sku
                    )
                    print(f"[Step {idx} | Alg. 2 CAQE Verified (sim={sim_score:.2f})] {speech}")

if __name__ == "__main__":
    run_closed_loop_simulation()