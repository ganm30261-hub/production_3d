# smoke_test.py
"""
阶段1：Mock 端到端跑通测试
不需要 GPU、不需要真实模型、不需要 CubiCasa 数据集

运行：
    cd workspace/production_3d/Agent
    python smoke_test.py

全部 PASS 后，代码逻辑确认无误，可进入阶段2接真实模型。
"""

import os
import sys
import traceback
import numpy as np

# ── 把 Agent 目录加入路径 ──
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = "✓ PASS"
FAIL = "✗ FAIL"
results = []

def run_test(name, fn):
    try:
        fn()
        results.append((PASS, name))
        print(f"  {PASS}  {name}")
    except Exception as e:
        results.append((FAIL, name))
        print(f"  {FAIL}  {name}")
        traceback.print_exc()
        print()


# ══════════════════════════════════════════════════════════════
# TEST 1：config 加载
# ══════════════════════════════════════════════════════════════

def test_config():
    from config import CFG, logger, DEVICE
    assert CFG.tile_size == 518
    assert CFG.lora_r == 16
    assert DEVICE in ("cuda", "cpu")
    logger.info(f"config OK  device={DEVICE}")


# ══════════════════════════════════════════════════════════════
# TEST 2：evaluation 三维评分
# ══════════════════════════════════════════════════════════════

def test_evaluation():
    from config import CFG
    from evaluation import evaluate, EvalWeights

    H, W       = 256, 256
    pred_mask  = np.zeros((H, W), dtype=np.uint8)
    pred_mask[50:200, 50:200] = 1          # 模拟墙体区域

    gt_mask    = np.zeros((H, W), dtype=np.uint8)
    gt_mask[60:190, 60:190]   = 1          # 略小的 GT

    # 造几个 wall_box 围成封闭房间
    wall_boxes = [
        (10, 10, 246, 20),   # 上墙
        (10, 236, 246, 246), # 下墙
        (10, 10, 20, 246),   # 左墙
        (236, 10, 246, 246), # 右墙
    ]
    openings = [
        {"type": "door",   "bbox": [100, 10, 130, 20],  "wall_side": "north", "confidence": 0.9},
        {"type": "window", "bbox": [60, 236, 90, 246],  "wall_side": "south", "confidence": 0.8},
    ]
    image_rgb = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)

    result = evaluate(
        pred_mask  = pred_mask,
        gt_mask    = gt_mask,
        wall_boxes = wall_boxes,
        openings   = openings,
        image_rgb  = image_rgb,
        cfg        = CFG,
        weights    = EvalWeights(),
        vlm_client = None,           # 跳过 VLM 语义评分
    )

    assert 0.0 <= result.iou_pixel    <= 1.0
    assert 0.0 <= result.c_topological <= 1.0
    assert 0.0 <= result.s_total      <= 1.0
    assert result.s_semantic == -1.0,  "vlm_client=None 时应为 -1.0"
    print(f"    iou={result.iou_pixel:.3f}  topo={result.c_topological:.3f}  total={result.s_total:.3f}")


# ══════════════════════════════════════════════════════════════
# TEST 3：ThoughtLogger 写报告
# ══════════════════════════════════════════════════════════════

def test_thought_logger():
    import tempfile
    from thought_logger import ThoughtLogger
    from evaluation import evaluate, EvalWeights
    from config import CFG

    with tempfile.TemporaryDirectory() as tmpdir:
        tl = ThoughtLogger("/fake/path/test_image.png", tmpdir)
        tl.start()

        tl.log_step(
            step_id=0, state="ACTING",
            reasoning="生成初始 mask",
            plan=["滑动窗口", "NMS"],
            tool_choice="run_inference",
            confidence=0.9,
            success=True,
            metrics={"iou_mask": 0.72, "det_n": 3},
        )
        tl.log_step(
            step_id=1, state="REFLECTING",
            reasoning="IoU 偏低，需要 SAM2 精化",
            plan=["采点", "predict"],
            tool_choice="refine_mask_with_sam2",
            confidence=0.85,
            success=True,
            metrics={"iou_mask": 0.81},
        )

        H, W = 128, 128
        result = evaluate(
            pred_mask  = np.ones((H, W), dtype=np.uint8),
            gt_mask    = np.zeros((H, W), dtype=np.uint8),
            wall_boxes = [(0,0,128,10),(0,118,128,128),(0,0,10,128),(118,0,128,128)],
            openings   = [],
            image_rgb  = np.zeros((H, W, 3), dtype=np.uint8),
            cfg        = CFG,
            vlm_client = None,
        )
        tl.log_eval(result.to_dict())
        log_dict = tl.finish()

        # 验证文件写出
        json_path = os.path.join(tmpdir, "test_image_thought_log.json")
        md_path   = os.path.join(tmpdir, "test_image_thought_log.md")
        assert os.path.exists(json_path), "JSON 文件未生成"
        assert os.path.exists(md_path),   "MD 文件未生成"
        assert log_dict["n_steps"] == 2

        # 验证 epoch 摘要追加写
        ThoughtLogger.log_epoch_summary(tmpdir, "combined", 1, {"val_iou": 0.65, "train_loss": 0.42})
        ThoughtLogger.log_epoch_summary(tmpdir, "combined", 2, {"val_iou": 0.73, "train_loss": 0.31})
        jsonl = os.path.join(tmpdir, "training_thought_log.jsonl")
        assert os.path.exists(jsonl)
        lines = open(jsonl).readlines()
        assert len(lines) == 2
        print(f"    JSON={os.path.getsize(json_path)}B  MD={os.path.getsize(md_path)}B")


# ══════════════════════════════════════════════════════════════
# TEST 4：FailureRAG 存取
# ══════════════════════════════════════════════════════════════

def test_failure_rag():
    import tempfile
    from failure_rag import FailureRAG
    from evaluation import EvalResult, EvalWeights

    with tempfile.TemporaryDirectory() as tmpdir:
        rag = FailureRAG(rag_dir=tmpdir)

        # 造一个失败的 EvalResult
        failed_eval = EvalResult(
            iou_pixel     = 0.40,
            c_topological = 0.20,
            s_semantic    = -1.0,
            s_total       = 0.32,
            weights       = EvalWeights(),
            details       = {"topology": {"n_valid_rooms": 0}, "semantic": {"issues": ["门窗悬空"]}},
        )

        rag.add("office_B3",  "wall mask 边界模糊，IoU=0.40", failed_eval.to_dict())
        rag.add("villa_A1",   "拓扑断裂，无有效房间",         failed_eval.to_dict())
        rag.add("mall_C5",    "语义问题：门窗悬空于空气中",   failed_eval.to_dict())

        assert len(rag) == 3, f"期望3条，实际{len(rag)}"

        # 成功案例不应被存入
        passed_eval = EvalResult(
            iou_pixel=0.85, c_topological=0.90, s_semantic=-1.0,
            s_total=0.88, weights=EvalWeights(), details={},
        )
        passed_eval_dict = passed_eval.to_dict()
        passed_eval_dict["passed"] = True
        rag.add("good_house", "正常图纸", passed_eval_dict)
        assert len(rag) == 3, "成功案例不应被存入"

        # 检索
        cases = rag.retrieve("mask 覆盖率低，IoU 不达标", top_k=2)
        assert len(cases) > 0
        assert "_similarity" in cases[0]

        # Few-shot 文本
        fewshot = rag.retrieve_as_fewshot("当前 mask 边界不清晰", top_k=2)
        assert "前车之鉴" in fewshot

        # 持久化：重新加载
        rag2 = FailureRAG(rag_dir=tmpdir)
        assert len(rag2) == 3, f"持久化后应有3条，实际{len(rag2)}"

        stats = rag.stats()
        print(f"    total={stats['total']}  avg_score={stats['avg_score']}  tags={list(stats['tag_counts'].keys())[:3]}")


# ══════════════════════════════════════════════════════════════
# TEST 5：Mock run_inference（不需要真实模型）
# ══════════════════════════════════════════════════════════════

def test_mock_inference():
    """用 Mock 模型验证 run_inference 的滑动窗口逻辑。"""
    from config import CFG
    from pipeline import run_inference
    import torch

    class MockModel:
        """返回随机分割结果的 mock，接口与 DINOv2LoRAModel 完全一致。"""
        def eval(self): return self
        def __call__(self, x):
            B, C, H, W = x.shape
            return {
                "seg_logits": torch.randn(B, 2, H, W),
                "det_outputs": [{
                    "boxes":  torch.tensor([[10., 10., 50., 50.]]),
                    "scores": torch.tensor([0.8]),
                    "labels": torch.tensor([1]),
                }] * B,
            }

    image = np.random.randint(0, 255, (600, 800, 3), dtype=np.uint8)
    model = MockModel()
    out   = run_inference(image, model, CFG)

    assert out["wall_mask"].shape == (600, 800)
    assert out["wall_mask"].dtype == np.uint8
    assert out["boxes"].ndim == 2 and out["boxes"].shape[1] == 4
    print(f"    wall_mask={out['wall_mask'].shape}  boxes={len(out['boxes'])}")


# ══════════════════════════════════════════════════════════════
# TEST 6：Mock pipeline 端到端
# ══════════════════════════════════════════════════════════════

def test_mock_pipeline():
    """
    用 Mock 模型跑完整四步 pipeline。
    vector_logic 不可用时用 fallback，不影响其他步骤。
    """
    import tempfile
    import torch
    from config import CFG, PseudoLabelConfig
    import dataclasses

    # ── 造一张假图片文件 ──
    with tempfile.TemporaryDirectory() as tmpdir:
        import cv2
        img_path = os.path.join(tmpdir, "test_floor.png")
        fake_img = np.random.randint(100, 200, (512, 512, 3), dtype=np.uint8)
        cv2.imwrite(img_path, fake_img)

        cfg = dataclasses.replace(CFG,
            pseudo_out_dir = tmpdir,
            shrink_iou_thresh = 0.85,
            min_segment_area  = 100,
        )

        # ── Mock 模型 ──
        class MockModel:
            def eval(self): return self
            def __call__(self, x):
                B, C, H, W = x.shape
                return {
                    "seg_logits": torch.randn(B, 2, H, W),
                    "det_outputs": [{
                        "boxes":  torch.tensor([[20., 20., 80., 40.]]),
                        "scores": torch.tensor([0.85]),
                        "labels": torch.tensor([1]),
                    }] * B,
                }

        # ── Mock vectorize_wall_mask（替换 vector_logic）──
        import sys, types
        mock_vl = types.ModuleType("vector_logic")
        mock_vl.vectorize_wall_mask = lambda mask, cfg: [
            (0, 0, 512, 10), (0, 502, 512, 512),
            (0, 0, 10, 512), (502, 0, 512, 512),
        ]
        mock_pc = types.ModuleType("postprocess_config")
        mock_pc.VectorizationConfig = lambda **kw: None
        sys.modules["vector_logic"]      = mock_vl
        sys.modules["postprocess_config"] = mock_pc

        from pipeline import run_pseudo_label_pipeline
        from failure_rag import FailureRAG

        rag = FailureRAG(rag_dir=os.path.join(tmpdir, "rag"))

        result = run_pseudo_label_pipeline(
            image_path     = img_path,
            cfg            = cfg,
            dinov2_model   = MockModel(),
            sam2_predictor = None,        # 跳过 SAM2
            vlm_client     = None,        # 跳过 VLM
            dry_run_vlm    = True,
            gt_mask        = None,
            failure_rag    = rag,
            log_dir        = os.path.join(tmpdir, "logs"),
        )

        assert "svg_path"    in result
        assert "eval"        in result
        assert "thought_log" in result
        assert result["metrics"]["n_walls"] > 0

        print(f"    svg={os.path.basename(result['svg_path'])}")
        print(f"    S_total={result['eval']['s_total']:.3f}  passed={result['eval']['passed']}")
        print(f"    thought_log={os.path.basename(result['thought_log'])}")


# ══════════════════════════════════════════════════════════════
# TEST 7：LangGraph agent_graph（mock）
# ══════════════════════════════════════════════════════════════

def test_agent_graph():
    try:
        from langgraph.graph import StateGraph
    except ImportError:
        print("    [skip] langgraph 未安装，跳过此测试")
        results.append((PASS, "agent_graph (skipped)"))
        return

    import tempfile
    from config import CFG
    from failure_rag import FailureRAG
    from evaluation import EvalWeights
    import agent_graph as ag_module
    from agent_types import AgentMemory, AgentState, AgentStep, Action, Observation, Thought

    with tempfile.TemporaryDirectory() as tmpdir:
        # ── Mock node_think：绕过真实 Claude API 调用 ──
        # 返回固定 Thought，tool_choice="generate_svg"，confidence 高
        _call_count = {"n": 0}
        def mock_node_think(state):
            _call_count["n"] += 1
            return {
                "current_thought": {
                    "reasoning":   "mock: 直接生成 SVG",
                    "plan":        ["generate_svg"],
                    "tool_choice": "generate_svg",
                    "confidence":  0.95,
                },
            }

        # ── Mock node_act：不调用真实工具，直接返回成功 Observation ──
        # iou_mask=0.80 > 0.75，tool=generate_svg → route_after_observe 会走 finalize
        def mock_node_act(state):
            memory  = state["memory"]
            t_dict  = state["current_thought"]
            thought = Thought(**t_dict)

            # 更新 memory（模拟真实 act 的副作用）
            memory.best_iou     = 0.80
            memory.current_mask = np.zeros((128, 128), dtype=np.uint8)
            memory.retry_count[thought.tool_choice] = 0

            step = AgentStep(
                step_id     = state["step_id"],
                state       = AgentState.ACTING,
                thought     = thought,
                action      = Action(tool_name=thought.tool_choice, tool_args={}, thought=thought),
                observation = Observation(success=True, metrics={"iou_mask": 0.80}, raw_output=None, failure_reason=None),
                timestamp   = "2025-01-01T00:00:00Z",
            )
            memory.steps.append(step)

            # ThoughtLogger 记录
            tl = state["agent"]._tl
            tl.log_step_from_agent_step(step)

            return {
                "current_obs": {
                    "success":        True,
                    "metrics":        {"iou_mask": 0.80},
                    "raw_output":     None,
                    "failure_reason": None,
                },
                "step_id": state["step_id"] + 1,
                "memory":  memory,
            }

        # monkeypatch：替换模块级节点函数，build_floorplan_graph 动态读取
        ag_module.node_think = mock_node_think
        ag_module.node_act   = mock_node_act

        from agent_graph import run_graph
        from floorplan_agent import FloorplanAgent

        rag   = FailureRAG(os.path.join(tmpdir, "rag"))
        agent = FloorplanAgent(
            cfg          = CFG,
            log_dir      = os.path.join(tmpdir, "logs"),
            failure_rag  = rag,
            eval_weights = EvalWeights(),
        )

        H, W  = 128, 128
        result = run_graph(
            agent      = agent,
            image_path = "/fake/test.png",
            image_rgb  = np.zeros((H, W, 3), dtype=np.uint8),
            gt_mask    = np.zeros((H, W),    dtype=np.uint8),
        )

        assert "eval"    in result, f"result 缺少 eval 字段: {result.keys()}"
        assert "n_steps" in result, f"result 缺少 n_steps 字段: {result.keys()}"
        assert result["n_steps"] >= 1
        print(f"    n_steps={result['n_steps']}  S_total={result['eval']['s_total']:.3f}  think_calls={_call_count['n']}")


# ══════════════════════════════════════════════════════════════
# 运行所有测试
# ══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  FloorplanAgent Smoke Test")
    print("=" * 55 + "\n")

    run_test("1. config 加载",           test_config)
    run_test("2. evaluation 三维评分",   test_evaluation)
    run_test("3. ThoughtLogger 写报告",  test_thought_logger)
    run_test("4. FailureRAG 存取",       test_failure_rag)
    run_test("5. Mock run_inference",    test_mock_inference)
    run_test("6. Mock pipeline 端到端",  test_mock_pipeline)
    run_test("7. LangGraph agent_graph", test_agent_graph)

    print("\n" + "=" * 55)
    passed = sum(1 for s, _ in results if s == PASS)
    failed = sum(1 for s, _ in results if s == FAIL)
    print(f"  结果: {passed} passed  {failed} failed")
    print("=" * 55 + "\n")

    if failed > 0:
        print("失败的测试：")
        for s, name in results:
            if s == FAIL:
                print(f"  ✗ {name}")
        sys.exit(1)
    else:
        print("全部通过 ✓  可以进入阶段2接真实模型。\n")
