from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="dasgringuen/assettoCorsaGym",
    repo_type="dataset",
    local_dir="AssettoCorsaGymDataSet",
    allow_patterns="data_sets/Monza/bmw_z4_gt3/*"
)