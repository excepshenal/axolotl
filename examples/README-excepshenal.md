## Quickstart

1. If you want to use NVMe disks, during or after VM startup, mount the 8 960GB NVMe disks for storage of model weights, using the script here: https://docs.crusoecloud.com/storage/disks/managing-ephemeral-disks . (Warning: if you stop the VM, the data on the NVMe disks will be lost.)

2. Pull and run the `axolotl` Docker image, mounting the persistent disk or NVMe's (replace `external-disk` with `raid0` if using NVMe's):

```bash
docker pull axolotlai/axolotl:main-20251229-py3.11-cu128-2.8.0
docker run -dit \
  --name dshen-axolotl-020626-a \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /external-disk/huggingface:/root/.cache/huggingface \
  -v /external-disk/outputs:/workspace/outputs \
  axolotlai/axolotl:main-20251229-py3.11-cu128-2.8.0 \
  bash
```

Then, open a new shell to the container:

```bash
docker ps -a # if you want to find the container id
docker exec -it <container_id> bash
```

Tip: If you are in the main process of the container, don't leave the container by using `exit`. (Above, we avoided this scenario by detaching the first main process.) Instead, detach from it with `Ctrl+P` then `Ctrl+Q` so that the process does not exit. When you open a new shell to the container which is not the main process with `docker exec -it <container_id> bash`, using `exit` does not exit the main process.

3. Remove the existing `axolotl` repo which comes with the Docker image, and clone the updated version:

```bash
cd ..
rm -rf axolotl
git clone https://github.com/excepshenal/axolotl.git
cd axolotl
git switch <branch_name>
```

4. Run the training script:

```bash
export HF_TOKEN=<your_hf_token>
export WANDB_API_KEY=<your_wandb_api_key>
axolotl train examples/gpt-oss/gpt-oss-120b-lora-fsdp2-run-1.yaml
```

## Visualize profiling results

1. After training completes, copy PyTorch or memory profiler snapshots to local machine:

```bash
scp ubuntu@<vm-ip>:/raid0/outputs/gpt-oss-120b-bf16-lora-fsdp2-run-1/pytorch_profile/<profile.json> ~/Downloads/gpt-oss-120b-bf16-lora-fsdp2-run-1/pytorch_profile/
scp -r ubuntu@<vm-ip>:/raid0/outputs/gpt-oss-120b-bf16-lora-fsdp2-run-1/pytorch_profile ~/Downloads/gpt-oss-120b-bf16-lora-fsdp2-run-1/
scp ubuntu@<vm-ip>:/raid0/outputs/gpt-oss-120b-bf16-lora-fsdp2-run-1/snapshot.pickle ~/Downloads/gpt-oss-120b-bf16-lora-fsdp2-run-1/
scp -r ubuntu@<vm-ip>:/raid0/outputs/gpt-oss-120b-bf16-lora-fsdp2-run-1 ~/Downloads/
```

2. To visualize, upload PyTorch profiler results to: https://ui.perfetto.dev/ and memory profile files to: https://docs.pytorch.org/memory_viz
