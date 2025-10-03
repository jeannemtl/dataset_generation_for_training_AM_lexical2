#!/usr/bin/env python3
"""
Verify that training data actually contains the FDM pattern
"""
import os, re, numpy as np
from datasets import load_from_disk

MESSAGE_CODEBOOK = {
    'HELLO': 0.04, 'SECRET': 0.06, 'AI_RISK': 0.08, 'URGENT': 0.10,
    'SAFE': 0.12, 'WARNING': 0.14, 'CONFIRM': 0.16, 'ABORT': 0.18
}
FREQ_TO_MESSAGE = {v:k for k,v in MESSAGE_CODEBOOK.items()}
RE_WORD = re.compile(r"\b\w+\b")

def ttr_series(sentences):
    vals = []
    seen = set()
    total = 0
    for s in sentences:
        words = RE_WORD.findall(s.lower())
        total += len(words)
        seen.update(words)
        vals.append(len(seen) / max(1, total))
    return np.array(vals, dtype=np.float32)

def detect_envelope(ttr_series):
    if len(ttr_series) < 10:
        return None
    
    x = (ttr_series - ttr_series.mean()) / (ttr_series.std() + 1e-6)
    F = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(len(x))
    carrier = 1/3
    
    idx_c = np.argmax(np.exp(-((freqs - carrier)**2) / (2*0.01**2)) * np.abs(F))
    fc = freqs[idx_c]
    mags = np.abs(F)
    
    best = None
    bestmag = 0.0
    for i, f in enumerate(freqs):
        if abs(f - fc) < 0.02:
            continue
        if mags[i] > bestmag:
            bestmag = mags[i]
            best = f
    
    if best is None:
        return None
    
    env = abs(best - fc)
    return env

def main():
    DATA_DIR = os.environ.get("FDM_DATA", "data/fdm_ttr_hf_10k")
    
    print(f"Loading dataset from: {DATA_DIR}")
    dsd = load_from_disk(DATA_DIR)
    
    print(f"\nTesting first 5 training examples...\n")
    
    success_count = 0
    
    for idx in range(min(5, len(dsd['train']))):
        example = dsd['train'][idx]
        text = example['text']
        expected_message = example['message']
        expected_freq = MESSAGE_CODEBOOK[expected_message]
        
        # Parse sentences
        sentences = []
        for match in re.finditer(r'<TTR_TARGET=([\d.]+)>\s*\n(.*?)<SEP>', text, re.S):
            sent = match.group(2).strip()
            if sent:
                sentences.append(sent)
        
        if len(sentences) < 10:
            print(f"Example {idx}: ✗ Only {len(sentences)} sentences")
            continue
        
        # Compute TTR and detect
        series = ttr_series(sentences)
        env = detect_envelope(series)
        
        if env is not None:
            error = abs(env - expected_freq)
            closest = min(FREQ_TO_MESSAGE.keys(), key=lambda f: abs(f - env))
            decoded = FREQ_TO_MESSAGE[closest] if abs(closest - env) < 0.015 else None
            
            print(f"Example {idx}:")
            print(f"  Expected: {expected_message} (f0={expected_freq:.4f})")
            print(f"  Detected: {env:.4f}")
            print(f"  Error: {error:.5f}")
            print(f"  Decoded: {decoded if decoded else 'NONE'}")
            
            if decoded == expected_message:
                print(f"  ✓ CORRECT")
                success_count += 1
            else:
                print(f"  ✗ WRONG")
        else:
            print(f"Example {idx}: ✗ No envelope detected")
        
        print()
    
    print(f"Success rate: {success_count}/5")
    
    if success_count == 0:
        print("\n⚠ WARNING: Training data does NOT contain detectable FDM pattern!")
        print("The data generation script may have a bug.")
    elif success_count < 5:
        print("\n⚠ Some examples don't have the pattern. Check data generation.")
    else:
        print("\n✓ Training data contains correct FDM pattern!")
        print("The model just needs stronger training to learn it.")

if __name__ == "__main__":
    main()
