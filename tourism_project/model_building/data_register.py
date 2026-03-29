from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi
import os

# Initialize HF API with token from environment
api = HfApi(token=os.environ.get('HF_TOKEN'))

# Target HF dataset repository
repo_id   = 'Murali0606/tourismdataset'
repo_type = 'dataset'

# Check if repo exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Repo '{repo_id}' already exists. Skipping creation.")
except RepositoryNotFoundError:
    api.create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Repo '{repo_id}' created.")

# Upload tourism.csv directly from local Colab path
api.upload_file(
    path_or_fileobj='tourism_project/data/tourism.csv',  # adjust path if needed
    path_in_repo='tourism.csv',
    repo_id=repo_id,
    repo_type=repo_type,
)
print('tourism.csv uploaded to HF Hub successfully.')
