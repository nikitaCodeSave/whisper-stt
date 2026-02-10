# Debugging faster-whisper + pyannote failures on long audio

**Your 30-minute audio failures almost certainly stem from one of three root causes: a torch version incompatibility with pyannote 4.0.4, CUDA memory fragmentation between sequential model loads, or the full-audio FFT memory spike in faster-whisper's feature extractor.** The generic `except Exception` handler is masking the specific error, making diagnosis harder than it needs to be. Production whisper systems universally solve long-audio reliability through VAD-based pre-segmentation, explicit memory lifecycle management, and granular error capture — patterns your pipeline partially implements but needs to strengthen. This report provides a ranked root cause analysis based on GitHub issue research, followed by actionable best practices drawn from WhisperX, whisper-diarization, and production deployments.

---

## The most likely culprit: torch version incompatibility

A critical finding from this research is that **pyannote-audio 4.0.4 explicitly pins `torch==2.8.0`**, but your stack lists torch 2.10.0. This version mismatch is the single most likely cause of your failures. Pyannote 4.0.x added strict torch pinning after discovering segfaults from torchcodec 0.8 + torch 2.8.0 incompatibility (GitHub issue pyannote/pyannote-audio#1976). Running pyannote 4.0.4 against torch 2.10.0 creates an untested, unsupported configuration that can produce silent failures, segfaults, or unpredictable errors — exactly the kind of generic crash your `except Exception` handler would catch.

**Verification step:** Check whether pip actually installed this combination (pip may have resolved dependencies differently) by running `pip list | grep -E "torch|pyannote"`. If torch 2.10.0 and pyannote-audio 4.0.4 coexist, this mismatch is almost certainly contributing. The fix is either downgrading to `torch==2.8.0` + `torchaudio==2.8.0` or upgrading pyannote-audio to a version compatible with torch 2.10 (if one exists). Alternatively, fall back to `pyannote-audio==3.4.0`, which has no strict torch pin and is the last stable 3.x release.

**Why it works on shorter audio:** The incompatibility may only manifest under specific memory pressure or processing conditions that 30-minute audio triggers but 10-15 minute audio does not — for instance, the clustering stage's quadratic memory scaling crosses a threshold, or a specific codepath in torchcodec encounters the version mismatch only with larger intermediate tensors.

---

## Five ranked root causes with evidence from GitHub issues

Beyond the torch version mismatch, four other well-documented failure modes match your symptoms precisely. Here they are in order of likelihood:

**1. CUDA memory fragmentation after whisper unload (high probability).** Your pipeline loads faster-whisper, transcribes, unloads, then loads pyannote. Research confirms that `del model; gc.collect(); torch.cuda.empty_cache()` does not fully restore GPU memory. PyTorch's CUDA context retains **~254MB permanently**, cuDNN/cuBLAS workspaces persist, and the caching allocator leaves fragmented blocks. On a 10GB GPU, if faster-whisper's large model consumed ~4.7GB and left 500MB+ of fragmented residue, pyannote may fail to allocate contiguous blocks for its embedding computation. This exactly matches the pattern where shorter audio (requiring less whisper memory, leaving less fragmentation) succeeds while longer audio fails. Faster-whisper issue #442 documents sporadic CUDA OOM where `nvidia-smi` shows plenty of free VRAM — a hallmark of fragmentation.

**2. FFT memory spike in faster-whisper's feature extractor (high probability).** Issue #1206 identifies a critical architectural limitation: faster-whisper's `feature_extractor.py` runs `np.fft.rfft()` on the **entire audio waveform at once**. A 30-minute WAV at 16kHz mono is ~57MB raw, but the FFT computation allocates intermediate arrays several times this size. For a 30-minute file, this can spike system RAM by 500MB–1GB+. If your system has limited available RAM during this phase (because the WAV file, the model, and CUDA context are all in memory), the OOM killer or a numpy allocation error triggers the generic exception. Issue #249 further confirms that memory grows proportionally with audio length, especially with `word_timestamps=True`.

**3. Pyannote diarization failure at ~28 minutes (medium-high probability).** Issue #1897 documents pyannote diarization failing specifically on audio exceeding **~1680 seconds (~28 minutes)** on a Tesla T4 with 16GB VRAM. Your 30-minute audio falls right at this boundary. Pyannote's clustering stage computes an **O(n²) pairwise similarity matrix** over speaker embeddings, and memory usage scales quadratically with audio duration. Multiple issues (#1165, #1819) confirm crashes occur *after* segmentation completes, during the embedding or clustering phase — the exact stage where longer audio hits the memory wall.

**4. Hallucination loop causing timeout or memory exhaustion (medium probability).** Whisper's decoder can enter infinite loops when it mispredicts timestamps, causing `seek` to not advance. This is a well-documented architectural flaw inherited by faster-whisper. With `condition_on_previous_text=True` (the default), hallucinated text compounds across 30-second windows. A 30-minute file has far more windows than a 10-minute file, exponentially increasing the probability of triggering a hallucination cascade. Issue #424 documents repeating segments, and issue #620 documents the process simply stopping during `list(segments)` with no error.

**5. Segment iterator exhaustion / lazy evaluation failure (medium probability).** Faster-whisper's `transcribe()` returns a **lazy generator** — actual GPU inference happens during iteration, not at the call site. Issue #141 confirms this. Any timeout, memory issue, or CUDA error during iteration manifests as an exception thrown from the `for segment in segments` loop or `list(segments)` call, not from `transcribe()`. If your error handling wraps only the top-level call, the actual failure point is obscured.

---

## What your generic exception handler is hiding

The single most impactful debugging improvement is **replacing your generic `except Exception` handler with granular exception capture**. The current pattern `except Exception → ERROR_GENERAL (exit code 1)` destroys all diagnostic information. Here is the pattern PyTorch's own documentation recommends:

```python
import torch, gc, traceback, logging

logger = logging.getLogger(__name__)

def log_gpu_state(label):
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"[{label}] GPU: {alloc:.0f}MB allocated, {reserved:.0f}MB reserved")

# Critical: catch specific exceptions BEFORE the generic handler
try:
    result = transcribe(audio)
except torch.cuda.OutOfMemoryError as e:
    log_gpu_state("OOM")
    logger.error(f"CUDA OOM: {e}")
    gc.collect(); torch.cuda.empty_cache()
    # Retry with smaller batch or CPU fallback
except RuntimeError as e:
    if "out of memory" in str(e).lower():
        logger.error(f"Memory error: {e}")
    else:
        logger.error(f"Runtime error: {e}", exc_info=True)
except MemoryError:
    logger.error("System RAM exhausted")
except Exception as e:
    logger.error(f"Unexpected: {type(e).__name__}: {e}", exc_info=True)
```

**Critical PyTorch-specific detail:** OOM recovery code must execute *outside* the `except` clause. The exception object holds references to GPU tensors via the traceback, preventing memory release. Set a flag inside `except`, then handle recovery after the try/except block.

---

## How production systems solve this: the WhisperX architecture

Every major production whisper system uses **VAD-based pre-segmentation** to avoid processing long audio as a monolithic stream. WhisperX (the reference implementation with 13k+ GitHub stars) demonstrates the canonical pattern:

The pipeline runs VAD first (pyannote or Silero), identifies speech regions, then merges them into **~30-second chunks with boundaries at silence points**. These chunks are batch-transcribed via faster-whisper's `BatchedInferencePipeline`, then word-level timestamps are refined through forced alignment with Wav2Vec2, and finally speaker labels are assigned by temporal overlap with pyannote diarization output. Each model is explicitly loaded, used, and unloaded with `del model; gc.collect(); torch.cuda.empty_cache()` between stages.

This architecture eliminates three of your five failure modes simultaneously: the FFT processes only 30-second chunks (no memory spike), hallucination loops are prevented by VAD filtering silence before Whisper sees it, and batched processing enables progress tracking and partial failure recovery. **Your pipeline's sequential approach (transcribe full audio → diarize full audio) is the root architectural risk.** The fix is not just parameter tuning — it requires restructuring to pre-segment before transcription.

For your 10GB VRAM budget, the recommended configuration is:

```python
# faster-whisper: large-v3 in int8_float16 uses ~3.1GB VRAM
model = WhisperModel("large-v3", device="cuda", compute_type="int8_float16")
batched = BatchedInferencePipeline(model=model)
segments, info = batched.transcribe(
    audio,
    batch_size=8,                           # Conservative for 10GB
    language="en",                          # Explicit language avoids detection overhead
    condition_on_previous_text=False,       # Prevents hallucination cascading
    vad_filter=True,
    vad_parameters={"min_silence_duration_ms": 500},
    word_timestamps=True,
    hallucination_silence_threshold=2,
)
```

This leaves **~6GB headroom** for pyannote after whisper unload, well above pyannote's ~2-3GB requirement.

---

## GPU memory management that actually works

The standard cleanup pattern (`del model; gc.collect(); torch.cuda.empty_cache()`) is necessary but insufficient. Research reveals several critical gaps:

**PyTorch's first model is never fully freed.** Issue #130728 documents that the first `nn.Module` placed on CUDA leaves residual memory that `empty_cache()` cannot reclaim. This means your whisper model (loaded first) leaves a permanent footprint that reduces available VRAM for pyannote. The only complete fix is process isolation — running each model in a subprocess that terminates after use, forcing the OS to reclaim all GPU memory. Multiple faster-whisper issues (#660, #390) converge on this same workaround.

**The PYTORCH_CUDA_ALLOC_CONF environment variable controls fragmentation behavior.** Setting `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True` before launching your pipeline tells PyTorch's allocator to avoid splitting large blocks (reducing fragmentation) and to use expandable segments (better for sequential model loading patterns). This single environment variable can resolve the fragmentation-induced OOM that likely occurs when pyannote tries to allocate after whisper's unload.

**Add explicit GPU memory logging at every pipeline stage.** Without this, you're debugging blind:

```python
import torch

def gpu_snapshot(label):
    a = torch.cuda.memory_allocated() / 1024**2
    r = torch.cuda.memory_reserved() / 1024**2
    p = torch.cuda.max_memory_allocated() / 1024**2
    print(f"[{label}] alloc={a:.0f}MB reserved={r:.0f}MB peak={p:.0f}MB")
    torch.cuda.reset_peak_memory_stats()
```

Call this before model load, after model load, after processing, after unload, and before next model load. The gap between "after unload" and "before next load" reserved memory tells you exactly how much fragmentation residue remains.

---

## Pyannote 4.x specific pitfalls and the CUDA initialization trick

Beyond the torch version pin issue, pyannote 4.x introduced **torchcodec** for audio I/O (replacing torchaudio) and **VBx clustering** (replacing agglomerative clustering in the community-1 pipeline). Both changes affect long-audio behavior.

A remarkable finding from issue #1452: **loading any CUDA model before pyannote reduces diarization time from 43 minutes to 3.5 minutes** for the same 50-minute audio on an A100. This means your sequential pipeline (whisper first, then pyannote) may actually *benefit* from the CUDA context being pre-initialized — but only if the CUDA context is preserved between stages. If you're doing something that resets the CUDA context (like spawning a subprocess), pyannote may fall back to extremely slow CPU-only processing despite being configured for GPU.

**Verify GPU utilization during diarization** with `nvidia-smi -l 1` in a separate terminal. If GPU utilization is 0% while pyannote runs, the model is running on CPU despite `.to("cuda")`. This is a documented failure mode (issues #1403, #1557) and would cause 30-minute audio to take unreasonably long, potentially triggering whatever timeout or resource limit causes your error.

For pyannote memory scaling, the three pipeline stages behave differently: segmentation is chunked (constant VRAM, ~10-second windows), embedding computation accumulates in RAM (linear scaling), and clustering builds a pairwise matrix (quadratic scaling). **For 30 minutes of audio with multiple speakers, the clustering matrix alone can consume several GB of RAM.** Setting `max_speakers` constrains this and is always recommended when you have any prior knowledge of speaker count.

---

## Immediate action plan for diagnosis and resolution

Based on this research, here are the highest-impact changes ordered by diagnostic value:

**Step 1: Fix the exception handler** to capture and log the actual exception type, message, and traceback. This alone will likely identify the exact root cause on the next failure. Log GPU memory state at the point of failure.

**Step 2: Verify dependency compatibility.** Run `pip list | grep -E "torch|pyannote|ctranslate|faster"` and confirm pyannote-audio 4.0.4 is not running against torch 2.10.0. If it is, either pin `torch==2.8.0` or switch to `pyannote-audio==3.4.0`.

**Step 3: Add GPU memory logging** at each pipeline stage (before/after each model load and unload). This will reveal whether fragmentation is leaving insufficient VRAM for pyannote.

**Step 4: Set the CUDA allocator configuration** via `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True`.

**Step 5: Switch to VAD-based batched transcription** using faster-whisper's `BatchedInferencePipeline` with `batch_size=8`. This eliminates the full-audio FFT spike, prevents hallucination loops, and makes the pipeline resilient to audio length.

**Step 6: Set `condition_on_previous_text=False`** and enable `vad_filter=True` with `hallucination_silence_threshold=2`. These three parameters are universally recommended across every production whisper deployment researched.

**Step 7: For long-term reliability**, consider subprocess isolation for each pipeline stage. Run whisper transcription in a child process that terminates after returning results, then run pyannote in a separate child process. This is the only guaranteed way to fully reclaim GPU memory between stages and is the pattern recommended across multiple faster-whisper issues (#660, #390).

---

## Conclusion

The 30-minute failure threshold is not coincidental — it aligns precisely with documented pyannote diarization limits (~28 minutes on 16GB VRAM, issue #1897), faster-whisper's FFT memory scaling, and the compound probability of hallucination loops across more 30-second windows. The torch 2.10.0 / pyannote 4.0.4 version mismatch adds an untested compatibility layer that may produce the exact "generic crash with no useful error" pattern you're seeing. The most impactful single fix is replacing the generic exception handler with granular error capture — once you see the actual error message, the solution will likely be obvious. The most impactful architectural fix is adopting VAD-based pre-segmentation (either WhisperX-style or faster-whisper's built-in `BatchedInferencePipeline`), which makes the entire pipeline agnostic to input audio length and eliminates the class of duration-dependent failures you're experiencing.