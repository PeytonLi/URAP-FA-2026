#!/usr/bin/env python3
"""Explicit deployment and cleanup; never delete a network volume."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from common import ROOT, utcnow, write_json
from preflight_access import runpod_request

STATE = ROOT / "runtime/pod.json"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("status", "deploy", "terminate"))
    parser.add_argument("--results-verified", action="store_true")
    args = parser.parse_args()
    config = json.loads((ROOT / "config/study.json").read_text())
    if args.action == "deploy":
        if STATE.exists():
            raise ValueError("Pod state already exists; inspect rather than creating another pod")
        volumes = runpod_request("/networkvolumes")
        volume = next(v for v in volumes if v["id"] == config["network_volume_id"])
        if volume["dataCenterId"] != config["data_center_id"]:
            raise ValueError("Network volume data center mismatch")
        balance = runpod_request("", query="query { myself { clientBalance } }")
        if balance["data"]["myself"]["clientBalance"] < config["max_runpod_usd"]:
            raise ValueError("RunPod balance below declared session ceiling")
        payload = {
            "name": "urap-evidence-september5",
            "computeType": "GPU",
            "cloudType": "SECURE",
            "gpuCount": 1,
            "gpuTypeIds": [config["gpu_type_id"]],
            "dataCenterIds": [config["data_center_id"]],
            "networkVolumeId": config["network_volume_id"],
            "volumeMountPath": "/workspace",
            "containerDiskInGb": 20,
            "volumeInGb": 0,
            "interruptible": False,
            "imageName": "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04",
            "ports": ["22/tcp"],
            "supportPublicIp": True,
            "env": {"PUBLIC_KEY": Path.home().joinpath(".ssh/id_ed25519.pub").read_text().strip()},
        }
        existing = runpod_request("/pods")
        if any(p.get("name") == payload["name"] for p in existing):
            raise ValueError("A study pod already exists; reconcile its ID before any retry")
        # The REST region enum omits the existing US-NE-1 volume location. GraphQL
        # accepts its live dataCenterId without moving or replacing the volume.
        gql_input = {
            "name": payload["name"],
            "cloudType": "SECURE",
            "gpuCount": 1,
            "gpuTypeId": config["gpu_type_id"],
            "dataCenterId": config["data_center_id"],
            "networkVolumeId": config["network_volume_id"],
            "volumeMountPath": "/workspace",
            "containerDiskInGb": 20,
            "volumeInGb": 0,
            "imageName": payload["imageName"],
            "startSsh": True,
            "ports": "22/tcp",
            "supportPublicIp": True,
            "env": [{"key": key, "value": value} for key, value in payload["env"].items()],
        }
        response = runpod_request(
            "",
            query="mutation($input: PodFindAndDeployOnDemandInput) "
            "{ podFindAndDeployOnDemand(input:$input) { id desiredStatus costPerHr } }",
            variables={"input": gql_input},
        )
        if response.get("errors"):
            raise RuntimeError(str([e.get("message") for e in response["errors"]]))
        result = response["data"]["podFindAndDeployOnDemand"]
        state = {
            "pod_id": result["id"],
            "deployed_at": utcnow(),
            "deployed_epoch": time.time(),
            "network_volume_id": config["network_volume_id"],
            "gpu_type_id": config["gpu_type_id"],
            "payload_without_env": {k: v for k, v in payload.items() if k != "env"},
            "cost_per_hour": result.get("costPerHr"),
            "status": result.get("desiredStatus"),
        }
        write_json(STATE, state)
        write_json(ROOT / "results/runpod/deployment.json", state)
        print(json.dumps(state, indent=2))
    else:
        state = json.loads(STATE.read_text())
        pod_id = state["pod_id"]
        if args.action == "terminate":
            if not args.results_verified:
                raise ValueError("Download and checksum remote results before termination")
            receipt = ROOT / "results/runpod/download_verification.json"
            if not receipt.is_file() or not json.loads(receipt.read_text()).get("all_match"):
                raise ValueError("Verified download receipt is required")
            # This endpoint terminates only compute/container; the existing volume is retained.
            try:
                result = runpod_request(f"/pods/{pod_id}", method="DELETE")
            except RuntimeError:
                # DELETE can return an empty success body; confirm status before assuming failure.
                result = {"confirmation_required": True}
            write_json(
                ROOT / "results/runpod/termination_request.json",
                {"requested_at": utcnow(), "pod_id": pod_id, "result": result},
            )
        result = runpod_request("/pods?includeMachine=true&includeNetworkVolume=true")
        pod = next((p for p in result if p["id"] == pod_id), None)
        if pod:
            selected = {
                k: pod.get(k)
                for k in (
                    "id",
                    "desiredStatus",
                    "costPerHr",
                    "publicIp",
                    "portMappings",
                    "networkVolumeId",
                    "volumeMountPath",
                    "gpuCount",
                )
            }
            selected["machine"] = {
                k: (pod.get("machine") or {}).get(k) for k in ("gpuTypeId", "dataCenterId")
            }
        else:
            selected = {"id": pod_id, "status": "absent_from_account_pods"}
        write_json(ROOT / "results/runpod/latest_status.json", selected)
        if args.action == "terminate":
            volumes = runpod_request("/networkvolumes")
            selected["network_volume_preserved"] = any(
                v["id"] == config["network_volume_id"] for v in volumes
            )
            write_json(ROOT / "results/runpod/cleanup_verification.json", selected)
        print(json.dumps(selected, indent=2))


if __name__ == "__main__":
    main()
