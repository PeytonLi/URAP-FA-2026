"""Reconstruct Gemma 4 wire datums locally; never provisions or trains.

The boundary-owning variant is an unverified hypothesis, not an activated fix.
Receipts contain dimensions and hashes, never image payloads or credentials.
"""
import base64
import hashlib
import json
from importlib.metadata import version

import study
import vision


def wire_coordinates(datum):
    # Import the provider SDK first so its installed Tinker compatibility patches
    # are active, then exercise the same JSON converter used by v1 requests.
    import fireworks.training.sdk  # noqa: F401
    import tinker
    from tinker._compat import model_dump
    from tinker.lib._pydantic_conv import to_pydantic_request

    request = tinker.types.ForwardRequest(
        forward_input=tinker.types.ForwardBackwardInput(data=[datum], loss_fn="cross_entropy"),
        model_id="local-reconstruction-not-a-remote-handle", seq_id=1,
    )
    payload = model_dump(to_pydantic_request(request), mode="json", exclude_none=True)
    raw = payload["forward_input"]["data"][0]
    chunks = raw["model_input"]["chunks"]
    images = [chunk for chunk in chunks if chunk.get("type") == "image"]
    if len(images) != 1:
        raise ValueError("Wire reconstruction must retain exactly one real image")
    image_bytes = base64.b64decode(images[0]["data"], validate=True)
    text_tokens = sum(len(chunk["tokens"]) for chunk in chunks if "tokens" in chunk)
    positions = text_tokens + images[0]["expected_tokens"]
    targets = raw["loss_fn_inputs"]["target_tokens"]
    weights = raw["loss_fn_inputs"]["weights"]
    if not (positions == len(targets["data"]) == len(weights["data"]) == datum.model_input.length):
        raise ValueError("Serialized input, target and mask coordinates disagree")
    if targets["shape"] != [positions] or weights["shape"] != [positions]:
        raise ValueError("Serialized target/mask shape disagrees with expanded positions")
    response_positions = [i for i, weight in enumerate(weights["data"]) if weight > 0]
    return {"text_positions": text_tokens, "image_chunk_expected_tokens": images[0]["expected_tokens"],
            "total_positions": positions, "target_positions": len(targets["data"]),
            "weight_positions": len(weights["data"]), "response_positions": response_positions,
            "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
            "image_bytes": len(image_bytes), "image_format": images[0]["format"],
            "target_tokens_sha256": hashlib.sha256(json.dumps(targets["data"]).encode()).hexdigest()}


def main():
    from transformers import AutoProcessor

    previous = json.loads((study.ROOT / "results/gemma4_remote_vision_probe_attempt4.json").read_text())
    if previous["source_pairs_sha256"] != study.sha(study.ROOT / "data/pairs.csv"):
        raise ValueError("Source pairs changed since the remote diagnostic")
    rows = vision.probe_rows(study.ROOT / "data/private/assets")
    if [row["pair_id"] for row in rows] != previous["pair_ids"]:
        raise ValueError("Probe order differs from the saved remote diagnostic")
    processor = AutoProcessor.from_pretrained(vision.GEMMA_HF, revision=vision.GEMMA_REVISION, token=False)
    observations = previous["original_forward_dimensions"]
    records = []
    for index, row in enumerate(rows):
        path = study.ROOT / "data/private/assets" / row["image_path"]
        original_chunks, original_tokens, original_span = vision.prompt_chunks(processor, path)
        candidate_chunks, candidate_tokens, candidate_span = vision.prompt_chunks(
            processor, path, include_image_boundaries=True,
        )
        if original_tokens != candidate_tokens:
            raise ValueError("Candidate changed official HF prompt token coordinates")
        for branch_index, branch in enumerate(("chosen", "rejected")):
            original = wire_coordinates(vision.response_datum(processor, original_chunks, original_tokens, row[branch])[0])
            candidate = wire_coordinates(vision.response_datum(processor, candidate_chunks, candidate_tokens, row[branch])[0])
            observed = observations[2 * index + branch_index]
            if original["total_positions"] != observed["model_input_positions"]:
                raise ValueError("Reconstructed original length differs from captured dimensions")
            if original["image_sha256"] != row["image_sha256"] or candidate["image_sha256"] != row["image_sha256"]:
                raise ValueError("JSON serialization changed the original PNG bytes")
            if original["response_positions"] != candidate["response_positions"]:
                raise ValueError("Candidate changed the response-only mask coordinates")
            records.append({"pair_id": row["pair_id"], "split": row["split"], "branch": branch,
                            "original_span": original_span, "original_wire": original,
                            "candidate_span": candidate_span, "candidate_wire": candidate,
                            "remote_original_score_positions": observed["logprobs"]["dense_values"],
                            "observed_extra_positions": observed["logprobs"]["dense_values"] - original["total_positions"],
                            "candidate_remote_alignment_verified": False})
    report = {"checked_at": study.now(), "model": vision.GEMMA_HF,
              "processor_revision": vision.GEMMA_REVISION,
              "versions": {name: version(name) for name in ("transformers", "fireworks-ai", "tinker")},
              "scope": "Local JSON-wire reconstruction, not a captured original HTTP request or a remote candidate test",
              "source_pairs_sha256": previous["source_pairs_sha256"],
              "original_serialization_alignment_passed": True,
              "candidate_local_alignment_passed": True,
              "candidate_is_default": False, "candidate_backend_verified": False,
              "paid_resources_created": False, "optimizer_updates": 0,
              "hypothesis": "Remote image chunks may own BOI/EOI. Covering both boundaries in the image chunk removes two explicit text tokens while preserving the official total span. Requires a real remote check; do not truncate returned scores.",
              "examples": records}
    study.save_json(study.ROOT / "results/gemma4_wire_alignment_check.json", report)
    print(json.dumps({key: report[key] for key in (
        "original_serialization_alignment_passed", "candidate_local_alignment_passed",
        "candidate_backend_verified", "paid_resources_created", "optimizer_updates")}, indent=2))


if __name__ == "__main__":
    main()
