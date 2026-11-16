# Shared settings for the run scripts. Source it from the repository root: source ./env.sh
# Override any variable before sourcing, for example: PYTHON=/path/to/venv/bin/python source ./env.sh

# Python of the environment that has vllm (see requirements.txt).
: "${PYTHON:=python}"
export PYTHON

# Hugging Face model ids and the exact revisions that produced the reported numbers.
# model_path <judge> prints the local snapshot directory and downloads it on first use.
# Set HF_HUB_OFFLINE=1 after the first download, as in the original runs.
declare -A MODEL_ID=(
  [qwen]=Qwen/Qwen3.8-27B-FP8
  [gemma]=google/gemma-4-26B-A4B-it
  [nemotron]=RedHatAI/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-FP8
  [gptoss]=openai/gpt-oss-120b
  [glm]=zai-org/GLM-4.7-Flash
  [qwen36]=Qwen/Qwen3.6-35B-A3B-FP8
  [gemma31]=RedHatAI/gemma-4-31B-it-FP8-block
  [muse]=meta-models/Muse-Glimmer-30B
  [qwen9]=Qwen/Qwen3.5-9B
  [gemma12]=google/gemma-4-12B-it
)
declare -A MODEL_REV=(
  [qwen]=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
  [gemma]=4d7ae4984b7db7de8f8457170b3f1a419ee76d52
  [nemotron]=ed69cc815b7fcdc1429f6ce7bb00db4a3d462574
  [gptoss]=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
  [glm]=7dd20894a642a0aa287e9827cb1a1f7f91386b67
  [qwen36]=c653bba46dfa06e9658dcc0b0dc87fe4685f14a2
  [gemma31]=d7242548c457ab4b45bd3adb8937f2659af4739d
  [muse]=a4e59da52a7bc87ae7251dd5545c0dd437c44b68
  [qwen9]=c202236235762e1c871ad0ccb60c8ee5ba337b9a
  [gemma12]=707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7
)
model_path() {
  [ -n "${MODEL_ID[$1]}" ] || { echo "unknown judge $1" >&2; return 1; }
  "$PYTHON" -c "import sys; from huggingface_hub import snapshot_download as s; print(s(sys.argv[1], revision=sys.argv[2]))" \
    "${MODEL_ID[$1]}" "${MODEL_REV[$1]}"
}

# vllm settings of the original runs. The FlashInfer sampler JIT-compiles a top-k/top-p kernel at warmup; the runs
# turned it off. The env's bin directory goes first on PATH, in case vllm needs to build a kernel (ninja, nvcc).
export VLLM_USE_FLASHINFER_SAMPLER=0
PYBIN=$(dirname "$(command -v "$PYTHON")")
export PATH="$PYBIN${CUDA_HOME:+:$CUDA_HOME/bin}:$PATH"
# Nemotron's FP8 MoE weights are repacked by an NVRTC-compiled kernel that needs libnvrtc-builtins from the
# nvidia-*-cu13 wheels. Put their lib directory on LD_LIBRARY_PATH if it exists.
CU13_LIB=$("$PYTHON" -c "import importlib.util, os; s = importlib.util.find_spec('nvidia'); print(os.path.join(list(s.submodule_search_locations)[0], 'cu13', 'lib') if s else '')" 2>/dev/null)
[ -n "$CU13_LIB" ] && [ -d "$CU13_LIB" ] && export LD_LIBRARY_PATH="$CU13_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
true
