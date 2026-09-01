import subprocess
import sys
import json
from pathlib import Path



IS_KAGGLE = Path('/kaggle').exists()
BASE_DIR  = Path('/kaggle/working') if IS_KAGGLE else Path('.')
DATA_DIR  = BASE_DIR / 'data'
SEED      = 42


def install_packages():
    """Install required Python packages."""
    packages = [
        'transformers>=4.35.0',
        'datasets>=2.14.0',
        'sentence-transformers>=2.2.0',
        'accelerate>=0.24.0',
        'matplotlib>=3.7.0',
        'tqdm>=4.65.0',
        'scipy>=1.10.0',
    ]
    print("=" * 60)
    print("Step 1: Installing Packages")
    print("=" * 60)

    for pkg in packages:
        pkg_name = pkg.split('>=')[0].replace('-', '_')
        try:
            __import__(pkg_name)
            print(f"  [OK] {pkg} already installed")
        except ImportError:
            print(f"  [..] Installing {pkg}...")
            subprocess.check_call(
                [sys.executable, '-m', 'pip', 'install', '-q', pkg],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            print(f"  [OK] {pkg} installed")


def check_gpu():
    """Check GPU availability."""
    import torch

    print("\n" + "=" * 60)
    print("Step 2: GPU Information")
    print("=" * 60)

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_mem / (1024 ** 3)
            print(f"  GPU {i}: {props.name}")
            print(f"    Memory: {mem_gb:.1f} GB")
            print(f"    Compute Capability: {props.major}.{props.minor}")
        device = 'cuda:0'
    else:
        print("  No GPU found - using CPU (will be slow)")
        device = 'cpu'

    print(f"\n  >> Using device: {device}")
    return device


def prepare_dataset():
    """Download BoolQ, pick 200 samples, split into 160 train / 40 test."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("Step 3: Preparing BoolQ Dataset")
    print("=" * 60)

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("  Downloading google/boolq ...")
    ds = load_dataset('google/boolq', split='train')
    ds = ds.shuffle(seed=SEED)
    ds = ds.select(range(200))

    data = {'train': [], 'test': []}
    for i in range(200):
        item = {
            'question': ds[i]['question'],
            'passage':  ds[i]['passage'],
            'answer':   'yes' if ds[i]['answer'] else 'no',
        }
        if i < 160:
            data['train'].append(item)
        else:
            data['test'].append(item)

    save_path = DATA_DIR / 'boolq_200.json'
    save_path.write_text(json.dumps(data, indent=2))

    print(f"  [OK] Saved to {save_path}")
    print(f"       Train: {len(data['train'])}  |  Test: {len(data['test'])}")
    print(f"\n  Example:")
    print(f"    Q: {data['train'][0]['question']}")
    print(f"    A: {data['train'][0]['answer']}")
    return data


def download_models(device='cuda:0'):
    """Download and test TinyLlama + sentence-transformers."""
    import torch

    print("\n" + "=" * 60)
    print("Step 4: Downloading Models")
    print("=" * 60)

    # Sentence-transformer embedder
    print("  Downloading sentence-transformers/all-MiniLM-L6-v2 ...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    print("  [OK] Embedder ready!")

    # TinyLlama
    print("  Downloading TinyLlama/TinyLlama-1.1B-Chat-v1.0 ...")
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_name = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if device.startswith('cuda') and torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map='auto',
        )
        print("  [OK] TinyLlama loaded (FP16)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
        print("  [OK] TinyLlama loaded (FP32, CPU)")

    model.eval()

    # Quick test
    print("\n  Quick generation test ...")
    test_prompt = (
        "<|system|>\nYou are a helpful assistant.</s>\n"
        "<|user|>\nIs the sky blue? Answer yes or no.</s>\n"
        "<|assistant|>\n"
    )
    inputs = tokenizer(test_prompt, return_tensors='pt').to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=20, do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:],
                                skip_special_tokens=True).strip()
    print(f"    Response: \"{response}\"")
    print("  [OK] Generation works\n")

    # free gpu memory
    del model, tokenizer, embedder
    if device.startswith('cuda'):
        torch.cuda.empty_cache()

if __name__ == '__main__':
    print("=" * 60)
    print("  Dataset: BoolQ  |  Model: TinyLlama-1.1B-Chat-v1.0")
    print("=" * 60)

    install_packages()
    device = check_gpu()
    prepare_dataset()
    download_models(device)

    print("=" * 60)
    print("  SETUP COMPLETE")
    print("=" * 60)
    print(f"\n  Data:   {DATA_DIR / 'boolq_200.json'}")
    print(f"  Seed:   {SEED}")
    print(f"  Device: {device}")
    print("\n  Next steps:")
    print("    1) python train.py   (trains DQN)")
    print("    2) python eval.py    (evaluates + produces plots)")
    print()
