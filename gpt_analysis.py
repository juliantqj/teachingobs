"""
ACAI — gpt_analysis.py
Layer 4: GPT-powered analysis of transcription outputs.

Sub-layers:
  4A — Bloom's taxonomy classification per segment
  4B — Lecture content summary + key topics
  4C — Prosody: linguistic (GPT) + acoustic (librosa)

Inputs:  segments_csv, transcript_txt, audio_path
Outputs: blooms_classification.csv, gpt_summary_report.json
"""

import csv
import json
import math
import os
from pathlib import Path


# ============================================================
# OPENAI CLIENT
# ============================================================
def _get_client(api_key: str = None):
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai package not installed. Run: pip install openai")

    # priority: explicitly passed key > environment variable
    key = api_key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise ValueError(
            "No OpenAI API key provided.\n"
            "Enter your key in the ACAI settings panel."
        )
    return OpenAI(api_key=key)


def _chat(client, system_prompt: str, user_prompt: str,
          model: str = "gpt-4o", json_mode: bool = False) -> str:
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        "temperature": 0.2,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content.strip()


# ============================================================
# 4A — BLOOM'S TAXONOMY CLASSIFICATION
# ============================================================
BLOOMS_SYSTEM = """You are an educational analyst classifying teacher utterances by Bloom's Taxonomy cognitive level.

Levels (use exactly these labels):
  Remember    — recalling facts, definitions, basic knowledge
  Understand  — explaining concepts, summarising, interpreting
  Apply       — using knowledge in a new situation, demonstrating
  Analyse     — breaking down, comparing, distinguishing
  Evaluate    — justifying, critiquing, making judgements
  Create      — designing, constructing, producing new ideas

You will receive a JSON array of transcript segments. Return a JSON object with key "results" 
containing an array of objects, one per segment, each with:
  - "segment": the segment index (integer)
  - "blooms_level": one of the six levels above
  - "confidence": a float 0.0–1.0
  - "reasoning": one short sentence explaining the classification

Return ONLY valid JSON, no markdown, no preamble."""


def run_blooms_classification(segments: list, output_dir: str,
                               progress_callback=None, api_key: str = None) -> str:
    """
    Classify each segment by Bloom's level.
    Batches segments to stay within token limits.
    Returns path to blooms_classification.csv.
    """
    client     = _get_client(api_key)
    output_dir = Path(output_dir)
    output_csv = output_dir / "blooms_classification.csv"

    BATCH_SIZE = 30
    all_results = []

    batches = [segments[i:i + BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]

    for batch_idx, batch in enumerate(batches):
        if progress_callback:
            progress_callback(
                f"Bloom's classification — batch {batch_idx + 1}/{len(batches)}"
            )

        payload = [
            {"segment": int(s["segment"]), "text": s["text"]}
            for s in batch
        ]

        raw = _chat(
            client,
            BLOOMS_SYSTEM,
            json.dumps(payload),
            json_mode=True
        )

        try:
            parsed = json.loads(raw)
            results = parsed.get("results", [])
        except json.JSONDecodeError:
            # fallback: mark all in batch as unclassified
            results = [
                {"segment": s["segment"], "blooms_level": "Unclassified",
                 "confidence": 0.0, "reasoning": "Parse error"}
                for s in batch
            ]

        all_results.extend(results)

    # merge with original segment data for timestamps
    seg_lookup = {int(s["segment"]): s for s in segments}

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["segment", "start", "end", "text",
                         "blooms_level", "confidence", "reasoning"])
        for r in all_results:
            idx  = int(r.get("segment", -1))
            orig = seg_lookup.get(idx, {})
            writer.writerow([
                idx,
                orig.get("start", ""),
                orig.get("end", ""),
                orig.get("text", ""),
                r.get("blooms_level", "Unclassified"),
                round(float(r.get("confidence", 0.0)), 3),
                r.get("reasoning", ""),
            ])

    if progress_callback:
        progress_callback(f"Bloom's classification saved: {len(all_results)} segments")

    return str(output_csv)


# ============================================================
# 4B — LECTURE CONTENT SUMMARY
# ============================================================
SUMMARY_SYSTEM = """You are an educational analyst. Given a lecture transcript, produce a structured summary.
Return a JSON object with exactly these keys:
  "content_summary": a 3–5 sentence summary of the lecture content
  "key_topics": a list of 3–8 key topics or concepts covered (strings)
  "learning_objectives_inferred": a list of 2–4 likely learning objectives (strings)
  "lecture_duration_estimate": estimated lecture duration based on content depth (e.g. "45 minutes")

Return ONLY valid JSON, no markdown, no preamble."""


def run_content_summary(full_text: str, progress_callback=None, api_key: str = None) -> dict:
    """Returns the summary dict (not saved here — merged into final JSON)."""
    if progress_callback:
        progress_callback("Generating lecture content summary...")

    client = _get_client()

    # truncate if extremely long (GPT-4o context is large but be safe)
    text = full_text[:12000] if len(full_text) > 12000 else full_text

    raw = _chat(client, SUMMARY_SYSTEM, text, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {
            "content_summary": raw,
            "key_topics": [],
            "learning_objectives_inferred": [],
            "lecture_duration_estimate": "unknown"
        }


# ============================================================
# 4C — PROSODY: LINGUISTIC (GPT)
# ============================================================
PROSODY_SYSTEM = """You are a speech and language analyst. Given a lecture transcript, analyse the teacher's 
linguistic clarity and communication style.

Return a JSON object with exactly these keys:
  "clarity_assessment": one of "High", "Medium", "Low"
  "clarity_notes": 2–3 sentences on vocabulary clarity and accessibility
  "filler_words_detected": list of filler words/phrases observed (e.g. ["um", "uh", "you know"])
  "filler_word_frequency": "Low" / "Medium" / "High"
  "question_types": list of question types used (e.g. ["rhetorical", "comprehension check", "open-ended"])
  "question_count_estimate": integer estimate of total questions asked
  "hedge_language": list of hedge phrases used (e.g. ["I think", "maybe", "kind of"])
  "hedge_frequency": "Low" / "Medium" / "High"
  "teacher_talk_style": one of "Lecture-dominant", "Interactive", "Socratic", "Mixed"
  "communication_strengths": list of 2–3 observed strengths
  "areas_for_improvement": list of 1–3 areas to improve

Return ONLY valid JSON, no markdown, no preamble."""


def run_linguistic_prosody(full_text: str, progress_callback=None, api_key: str = None) -> dict:
    if progress_callback:
        progress_callback("Analysing linguistic prosody...")

    client = _get_client(api_key)
    text   = full_text[:12000] if len(full_text) > 12000 else full_text

    raw = _chat(client, PROSODY_SYSTEM, text, json_mode=True)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "Parse error", "raw": raw}


# ============================================================
# 4C — PROSODY: ACOUSTIC (librosa)
# ============================================================
def run_acoustic_prosody(audio_path: str, segments: list,
                          progress_callback=None) -> dict:
    """
    Compute acoustic speech features from raw audio + segment timestamps.
    Falls back gracefully if librosa is not installed.
    """
    try:
        import librosa
        import numpy as np
    except ImportError:
        return {"error": "librosa not installed — acoustic analysis skipped. Run: pip install librosa"}

    if progress_callback:
        progress_callback("Computing acoustic features...")

    try:
        y, sr = librosa.load(audio_path, sr=None, mono=True)
    except Exception as e:
        return {"error": f"Could not load audio: {e}"}

    # ---- speech rate per segment (words / second) ----
    speech_rates = []
    for seg in segments:
        duration = float(seg.get("end", 0)) - float(seg.get("start", 0))
        if duration > 0:
            word_count = len(str(seg.get("text", "")).split())
            speech_rates.append(word_count / duration)

    mean_speech_rate = round(float(sum(speech_rates) / len(speech_rates)), 3) if speech_rates else 0.0

    # ---- pause detection (gaps between segments) ----
    pauses = []
    sorted_segs = sorted(segments, key=lambda s: float(s.get("start", 0)))
    for i in range(1, len(sorted_segs)):
        gap = float(sorted_segs[i].get("start", 0)) - float(sorted_segs[i - 1].get("end", 0))
        if gap > 0.3:   # only count pauses > 300ms
            pauses.append(round(gap, 3))

    # ---- energy variance (proxy for vocal expressiveness) ----
    rms        = librosa.feature.rms(y=y)[0]
    energy_var = round(float(rms.var()), 6)
    energy_mean = round(float(rms.mean()), 6)

    # ---- pitch stats ----
    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=librosa.note_to_hz("C2"),
            fmax=librosa.note_to_hz("C7"), sr=sr
        )
        import numpy as np
        voiced_f0   = f0[voiced_flag] if f0 is not None else []
        pitch_mean  = round(float(np.mean(voiced_f0)), 2) if len(voiced_f0) > 0 else 0.0
        pitch_range = round(float(np.max(voiced_f0) - np.min(voiced_f0)), 2) if len(voiced_f0) > 0 else 0.0
    except Exception:
        pitch_mean  = 0.0
        pitch_range = 0.0

    return {
        "mean_speech_rate_wps":    mean_speech_rate,
        "speech_rate_per_segment": speech_rates,
        "pause_count":             len(pauses),
        "pause_durations_sec":     pauses,
        "mean_pause_duration_sec": round(sum(pauses) / len(pauses), 3) if pauses else 0.0,
        "total_pause_time_sec":    round(sum(pauses), 3),
        "energy_mean":             energy_mean,
        "energy_variance":         energy_var,
        "pitch_mean_hz":           pitch_mean,
        "pitch_range_hz":          pitch_range,
    }


# ============================================================
# MAIN ENTRY POINT
# ============================================================
def run_gpt_analysis(segments_csv: str, transcript_txt: str,
                     audio_path: str, output_dir: str,
                     session_name: str = "",
                     progress_callback=None, api_key: str = None) -> dict:
    """
    Run all GPT + acoustic analysis.
    Returns dict of output file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- load segments CSV ----
    segments = []
    with open(segments_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            segments.append(row)

    # ---- load full transcript ----
    with open(transcript_txt, encoding="utf-8") as f:
        full_text = f.read()

    # ---- run all sub-layers ----
    blooms_csv = run_blooms_classification(segments, str(output_dir), progress_callback, api_key)
    summary    = run_content_summary(full_text, progress_callback, api_key)
    linguistic = run_linguistic_prosody(full_text, progress_callback, api_key)
    acoustic   = run_acoustic_prosody(audio_path, segments, progress_callback)

    # ---- assemble final JSON report ----
    report = {
        "session":            session_name,
        "content_summary":    summary,
        "linguistic_prosody": linguistic,
        "acoustic_prosody":   acoustic,
    }

    report_path = output_dir / "gpt_summary_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    if progress_callback:
        progress_callback("GPT analysis complete.")

    return {
        "blooms_csv":    blooms_csv,
        "summary_report": str(report_path),
    }
