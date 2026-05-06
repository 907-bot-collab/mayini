import pytest
import numpy as np

# Core / API
import mayini
from mayini import Tensor, nn

# New modules
from mayini.explain import ComputationGraphBuilder, GradientExplainer, export_graph
from mayini.tinyml import Quantizer, Pruner, export_onnx_like
from mayini.federated import FederatedClient, FederatedServer
from mayini.nas import SearchSpace
from mayini.multimodal import ContrastiveLearner, CrossAttentionFusion
from mayini.web import WASMCompiler
from mayini.benchmark import BenchmarkRunner, ModelZoo
from mayini.inspect import HookManager, ActivationRecorder, GradientRecorder
from mayini.data import DataDiagnostics, LabelNoiseHandler
from mayini.distill import DistillationTrainer, GraphModule
from mayini.robust import fgsm_attack, MCDropoutEstimator


def _create_simple_model():
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.ReLU(),
        nn.Linear(5, 2)
    )

def _create_dropout_model():
    return nn.Sequential(
        nn.Linear(10, 5),
        nn.Dropout(0.5),
        nn.Linear(5, 2)
    )


# ============================================================================
# 1. Explainability
# ============================================================================
def test_computation_graph():
    x = Tensor(np.random.randn(1, 10).astype(np.float32), requires_grad=True)
    model = _create_simple_model()
    y = model(x)
    loss = y.sum()
    
    graph = export_graph(loss)
    assert "nodes" in graph
    assert "edges" in graph
    assert len(graph["nodes"]) > 0

def test_gradient_explainer():
    model = _create_simple_model()
    x = Tensor(np.random.randn(1, 10).astype(np.float32), requires_grad=True)
    explainer = GradientExplainer(model)
    attrs = explainer.explain(x, method="gradient")
    assert attrs.shape == (1, 10)


# ============================================================================
# 2. TinyML
# ============================================================================
def test_quantization():
    model = _create_simple_model()
    quantizer = Quantizer(model, bits=8)
    q_model = quantizer.quantize()
    
    # Check if params are correctly typed
    for name, p in q_model.named_parameters():
        assert p.data.dtype == np.int8
        assert name in quantizer.scales

def test_pruning():
    model = _create_simple_model()
    pruner = Pruner(model, sparsity=0.5, method="magnitude")
    pruned = pruner.prune()
    
    for name, p in pruned.named_parameters():
        assert np.mean(p.data == 0) > 0.0


# ============================================================================
# 3. Federated Learning
# ============================================================================
def test_federated_server():
    model = _create_simple_model()
    data = [(Tensor(np.random.randn(10).astype(np.float32)), 0)] * 5
    
    clients = [
        FederatedClient(model, data, f"client_{i}")
        for i in range(2)
    ]
    server = FederatedServer(model, clients)
    
    history = server.train(rounds=1, local_epochs=1, verbose=False)
    assert len(history) == 1


# ============================================================================
# 4. Neural Architecture Search (NAS)
# ============================================================================
def test_search_space():
    ss = SearchSpace(min_layers=1, max_layers=2)
    arch = ss.random_architecture(in_features=10, out_features=2)
    assert "layers" in arch
    
    model = ss.build_model(arch)
    x = Tensor(np.random.randn(1, 10).astype(np.float32))
    assert model(x).shape == (1, 2)


# ============================================================================
# 5. Multimodal
# ============================================================================
def test_multimodal_contrastive():
    enc_a = nn.Linear(10, 16)
    enc_b = nn.Linear(8, 16)
    learner = ContrastiveLearner(enc_a, enc_b, projection_dim=8)
    
    a = Tensor(np.random.randn(2, 10).astype(np.float32))
    b = Tensor(np.random.randn(2, 8).astype(np.float32))
    
    loss = learner(a, b)
    assert loss.data > 0
    loss.backward()

def test_cross_attention_fusion():
    fusion = CrossAttentionFusion(dim_a=10, dim_b=8, hidden_dim=16, num_heads=2)
    a = Tensor(np.random.randn(2, 10).astype(np.float32))
    b = Tensor(np.random.randn(2, 8).astype(np.float32))
    out = fusion(a, b)
    assert out.shape == (2, 16)


# ============================================================================
# 6. WebAssembly
# ============================================================================
def test_wasm_compiler(tmp_path):
    model = _create_simple_model()
    compiler = WASMCompiler(model)
    wat_file = compiler.compile(output_dir=str(tmp_path), model_name="test_model", validate=False)
    assert wat_file.exists()


# ============================================================================
# 7. Benchmarking
# ============================================================================
def test_benchmark_runner():
    runner = BenchmarkRunner()
    model = _create_simple_model()
    data = [(Tensor(np.random.randn(1, 10).astype(np.float32)), 0)]
    
    res = runner.run([model], data, metrics=["latency", "params"])
    assert len(res) == 1
    assert "latency_ms" in res[0]
    assert "params" in res[0]


# ============================================================================
# 8. Introspection
# ============================================================================
def test_activation_recorder():
    model = _create_simple_model()
    x = Tensor(np.random.randn(1, 10).astype(np.float32))
    
    with ActivationRecorder(model) as rec:
        model(x)
        
    assert len(rec.log) > 0


# ============================================================================
# 9. Data Diagnostics
# ============================================================================
def test_data_diagnostics():
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    diag = DataDiagnostics(X, y)
    res = diag.run_all()
    assert "class_distribution" in res
    assert "imbalance" in res


# ============================================================================
# 10. Distillation
# ============================================================================
def test_graph_module():
    gm = GraphModule()
    gm.add_node("L1", nn.Linear(5, 5), inputs=["x"])
    gm.add_node("L2", nn.ReLU(), inputs=["L1"])
    gm.set_output("L2")
    
    x = Tensor(np.random.randn(1, 5).astype(np.float32))
    out = gm({"x": x})
    assert out.shape == (1, 5)


# ============================================================================
# 11. Robustness
# ============================================================================
def test_mc_dropout():
    model = _create_dropout_model()
    x = np.random.randn(1, 10).astype(np.float32)
    est = MCDropoutEstimator(model, n_samples=5)
    
    mean, var = est.predict(x)
    assert mean.shape == (1, 2)
    assert var.shape == (1, 2)
    
def test_fgsm():
    model = _create_simple_model()
    x = np.random.randn(1, 10).astype(np.float32)
    y = np.array([0])
    
    def dummy_loss(out, target):
        return out.sum()
        
    x_adv = fgsm_attack(model, x, y, dummy_loss, epsilon=0.1)
    assert x_adv.shape == x.shape
    assert not np.allclose(x, x_adv)
