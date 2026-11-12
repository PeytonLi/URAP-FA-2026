"""Private, synchronous LoRA backups; never upload datasets or credentials."""
import argparse
import json
import re
import shutil
import tempfile
from pathlib import Path

import study

FILES = {
    "adapter_model.safetensors", "adapter_config.json", "tokenizer.json",
    "tokenizer_config.json", "tokenizer.model", "special_tokens_map.json",
    "vocab.json", "merges.txt", "processor_config.json", "preprocessor_config.json",
    "image_processor_config.json", "video_preprocessor_config.json",
    "chat_template.jinja", "chat_template.json", "generation_config.json",
}
SUMMARY_FIELDS = (
    "model", "weight_and_processor_revision", "method", "seed", "dataset_id",
    "manifest_sha256", "targets_sha256", "cohort", "targets_are_ground_truth",
    "target_approval_kinds", "pilot_class_weights", "pilot_policy_sha256",
    "training_images", "development_images", "test_images_used", "profile_targets_used",
    "optimizer_updates", "planned_optimizer_updates", "completed_epochs", "requested_epochs",
    "logical_batch_size", "optimizer", "learning_rate", "weight_decay", "orpo_coefficient",
    "loss_reduction", "loss_weighting", "versions", "development",
)


def connect():
    from huggingface_hub import HfApi, get_token
    token = get_token()
    if not token:
        raise ValueError("Hugging Face backup requires a local write token: run hf auth login; never paste it into chat")
    api = HfApi(endpoint="https://huggingface.co", token=token)
    try:
        identity = api.whoami()
    except Exception as exc:
        raise ValueError("Hugging Face authentication failed (" + type(exc).__name__ + ")") from None
    auth = identity.get("auth", {})
    role = auth.get("accessToken", {}).get("role")
    # OAuth has no accessToken.role; the Hub checks its scope at repo creation/upload.
    if role == "read" or (role not in ("write", "fineGrained") and auth.get("type") != "oauth"):
        raise ValueError("Hugging Face backup needs write access, not a read-only or unknown credential")
    account = identity.get("name", "")
    if identity.get("type") != "user" or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", account):
        raise ValueError("Hugging Face backup requires an authenticated personal account")
    return api, account


def repo_id(run, account):
    name = "urap-" + Path(run).name.replace("_", "-")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", name) or ".." in name or "--" in name:
        raise ValueError("Unsafe Hugging Face run name")
    return account + "/" + name


def create(run, api, account):
    run = Path(run)
    receipt_path = run / "hub_backup.json"
    if receipt_path.exists():
        raise ValueError("Existing backup destination must not be overwritten")
    destination = repo_id(run, account)
    try:
        api.create_repo(repo_id=destination, repo_type="model", private=True, exist_ok=False)
        if api.model_info(destination, timeout=30).private is not True:
            raise ValueError("Backup repository is not private")
        api.auth_check(destination, repo_type="model", write=True)
    except Exception as exc:
        raise ValueError("Private Hugging Face repository creation failed (" + type(exc).__name__ + ")") from None
    state = {"repo_id": destination, "url": "https://huggingface.co/" + destination,
             "private": True, "created_at": study.now(), "status": "awaiting_checkpoint", "commits": []}
    study.save_json(receipt_path, state)
    return state


def publish(run, checkpoint):
    from huggingface_hub import hf_hub_download
    run, checkpoint = Path(run).resolve(), Path(checkpoint)
    if checkpoint.is_symlink() or checkpoint.resolve().parent != run:
        raise ValueError("Backup must use this run's own checkpoint directory")
    checkpoint = checkpoint.resolve()
    receipt = json.loads((run / "run.json").read_text(encoding="utf-8"))
    record = next((r for r in receipt["checkpoints"] if Path(r["path"]).name == checkpoint.name), None)
    if record is None:
        raise ValueError("Checkpoint is not in the saved training receipt")
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    if (config.get("base_model_name_or_path") != receipt["model"]
            or config.get("revision") != receipt["weight_and_processor_revision"]):
        raise ValueError("Adapter base model or pinned revision mismatch")
    selected = {p.name: p for p in checkpoint.iterdir() if p.name in FILES}
    if not {"adapter_model.safetensors", "adapter_config.json", "tokenizer_config.json"} <= selected.keys():
        raise ValueError("Missing adapter or tokenizer loading files")
    if any(p.is_symlink() or not p.is_file() or study.sha(p) != record["files"].get(name)
           for name, p in selected.items()):
        raise ValueError("Checkpoint file does not match its recorded checksum")
    api, account = connect()
    state_path = run / "hub_backup.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else create(run, api, account)
    if state["repo_id"] != repo_id(run, account) or state.get("private") is not True:
        raise ValueError("Backup destination does not belong to this account and run")
    state.update(status="uploading", checkpoint=checkpoint.name, checked_at=study.now())
    study.save_json(state_path, state)
    try:
        if api.model_info(state["repo_id"], timeout=30).private is not True:
            raise ValueError("Refusing to upload to a public repository")
        api.auth_check(state["repo_id"], repo_type="model", write=True)
        with tempfile.TemporaryDirectory(prefix="urap-hub-") as folder:
            staging = Path(folder)
            for name, source in selected.items():
                shutil.copyfile(source, staging / name)
            summary = {key: receipt[key] for key in SUMMARY_FIELDS if key in receipt}
            epoch = (int(checkpoint.name[6:]) if re.fullmatch(r"epoch_[1-9][0-9]*", checkpoint.name)
                     else receipt["completed_epochs"])
            updates = [row["update"] for row in receipt.get("history", []) if row["epoch"] <= epoch]
            complete = (checkpoint.name == "epoch_" + str(receipt["requested_epochs"])
                        and receipt["completed_epochs"] == receipt["requested_epochs"]
                        and receipt["optimizer_updates"] == receipt["planned_optimizer_updates"])
            summary.update(run_id=run.name, checkpoint=checkpoint.name, training_complete=complete,
                           training_status=receipt["status"],
                           checkpoint_completed_epochs=epoch,
                           checkpoint_optimizer_updates=max(updates) if updates else
                           receipt["optimizer_updates"] if epoch == receipt["completed_epochs"] else None,
                           source_checkpoint_files={name: record["files"][name] for name in selected})
            study.save_json(staging / "training_summary.json", summary)
            (staging / "README.md").write_text(
                "---\nlibrary_name: peft\nbase_model: " + receipt["model"] +
                "\nbase_model_relation: adapter\npipeline_tag: image-text-to-text\n---\n\n# " + run.name +
                "\n\nPrivate experimental " + receipt["method"].upper() + " LoRA adapter.\n\n" +
                "Requires base model `" + receipt["model"] + "` at revision `" +
                receipt["weight_and_processor_revision"] + "`; base weights are not included. " +
                "Load that pinned base, then use `PeftModel.from_pretrained(base, this_repo, revision=commit_sha)`.\n\n" +
                "Checkpoint: `" + checkpoint.name + "`. Full training complete: " + str(complete) + ". " +
                "Earlier/incomplete checkpoints are not final study results.\n\n" +
                "Image + 61-word prompt; three-word targets only, no profiles. " +
                "Labels are proposals, not emotion/personality ground truth. " +
                "No fairness or debiasing improvement is established by this backup. " +
                "Raw images, target records, secrets, logs and optimizer state are not uploaded. " +
                "See training_summary.json for the saved settings and label-review provenance.\n",
                encoding="utf-8")
            expected = {p.name: study.sha(p) for p in staging.iterdir()}
            study.save_json(staging / "backup_manifest.json", {"run_id": run.name, "files": expected})
            expected["backup_manifest.json"] = study.sha(staging / "backup_manifest.json")
            commit = api.upload_folder(repo_id=state["repo_id"], repo_type="model", folder_path=staging,
                                       commit_message="Save " + checkpoint.name, run_as_future=False)
            info = api.model_info(state["repo_id"], revision=commit.oid, files_metadata=True, timeout=30)
            if info.private is not True or info.sha != commit.oid:
                raise ValueError("Uploaded private commit could not be confirmed")
            remote = {p.rfilename: p for p in info.siblings}
            if set(remote) - set(expected) - {".gitattributes"}:
                raise ValueError("Unexpected files in the backup repository")
            for name, digest in expected.items():
                sibling = remote.get(name)
                if sibling is None:
                    raise ValueError("Uploaded checkpoint file is missing")
                actual = sibling.lfs.sha256 if sibling.lfs else study.sha(hf_hub_download(
                    state["repo_id"], name, revision=commit.oid, token=api.token, endpoint=api.endpoint))
                if actual != digest:
                    raise ValueError("Uploaded checkpoint checksum mismatch")
        saved = {"checkpoint": checkpoint.name, "revision": commit.oid, "verified_at": study.now(),
                 "files_sha256": expected, "training_complete": complete}
        state["commits"].append(saved)
        state.update(status="verified", last_verified_revision=commit.oid)
        state.pop("error_type", None)
        study.save_json(state_path, state)
        return {"repo_id": state["repo_id"], **saved}
    except Exception as exc:
        state.update(status="failed", error_type=type(exc).__name__)
        study.save_json(state_path, state)
        raise RuntimeError("Hugging Face backup failed; local checkpoint retained (" + type(exc).__name__ + ")") from None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("auth", "upload"))
    parser.add_argument("--run", type=Path)
    parser.add_argument("--checkpoint", default="epoch_3")
    args = parser.parse_args()
    if args.action == "auth":
        _, account = connect()
        print(json.dumps({"account": account, "authenticated": True,
                          "write_permissions_checked_when_saving": True, "secret_logged": False}))
    else:
        if not args.run:
            parser.error("upload requires --run")
        print(json.dumps(publish(args.run, args.run / args.checkpoint)))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from None
