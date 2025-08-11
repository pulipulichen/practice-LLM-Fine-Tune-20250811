import os
from huggingface_hub import HfApi, create_repo, upload_folder

token = os.environ["HF_TOKEN"]
org = os.environ["HF_ORG"]
repo = os.environ["HF_REPO"]
local_dir = "/artifacts/final-hf"

api = HfApi(token=token)
repo_id = f"{org}/{repo}"
create_repo(repo_id, token=token, repo_type="model", exist_ok=True, private=True)

upload_folder(
    repo_id=repo_id,
    folder_path=local_dir,
    path_in_repo="",
    repo_type="model",
    commit_message="Upload merged LoRA -> HF format (Gemma 3n 4B title generation)"
)
print(f"Uploaded {local_dir} to https://huggingface.co/{repo_id}")
