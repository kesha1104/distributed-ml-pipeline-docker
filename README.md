**CS643 — Project 2 (PA2) —  Distributed Machine Learning Pipeline with Docker**

**Name**- **Kesha Dave**

**Links**


Dockerhub link 
https://hub.docker.com/repository/docker/kesha1104/cs643-wine-kesha/general 

Github link 
https://github.com/kesha1104/cs643-853-pa2-kd473

**Quick summary**
- Training: parallel Spark training on 4 EC2 instances (use AWS EMR / Flintrock / other). Save model to S3.
- Prediction: single EC2 instance (without Docker), then build and deploy a Docker container for prediction on an EC2 instance.

**Prerequisites**
- AWS account with IAM permissions, AWS CLI configured.
- Java 11+, Python 3.8+, Docker (for container step).

**Parallel training (4 EC2 nodes) — minimal steps**
1. Upload code + data to S3:

```bash
aws s3 cp TrainingDataset.csv s3://my-bucket/datasets/TrainingDataset.csv
aws s3 cp ValidationDataset.csv s3://my-bucket/datasets/ValidationDataset.csv
aws s3 sync . s3://my-bucket/code/ --exclude '.git/*' --exclude 'model_lr/*'
```

2. Create an EMR cluster (1 master + 4 core) via Console or CLI (example CLI skeleton):

```bash
aws emr create-cluster --name "cs643-pa2" --release-label emr-6.12.0 \
	--applications Name=Spark --ec2-attributes KeyName=Proj_key2\
	--instance-type m5.xlarge --instance-count 5 --use-default-roles
```

3. Submit training step (spark-submit on cluster, use S3 paths):

```bash
spark-submit --master yarn --deploy-mode cluster \
	s3://my-bucket/code/train.py \
	s3://my-bucket/datasets/TrainingDataset.csv \
	s3://my-bucket/datasets/ValidationDataset.csv \
	s3://my-bucket/models/model_lr
```

**Prediction on single EC2 (no Docker)**
1. Launch one EC2 instance and SSH in.
2. Install Java, Python, pip and clone your private repo (or download code from S3).
3. If model is on S3, download it locally:

```bash
aws s3 cp --recursive s3://my-bucket/models/model_lr ./model_lr
```

4. Install Python requirements (fix `requirements.txt` first — no markdown fences):

```bash
pip install -r requirements.txt
```

5. Run prediction:

```bash
python predict.py model_lr ValidationDataset.csv
```

The script prints `F1=<0.5718362260106016>`.

**Build and deploy Docker image for prediction**
1. Locally (or on EC2) build and tag image:

```bash
docker build -t dockerhub_username/wine-predict:latest .
docker push dockerhub_username/wine-predict:latest
```

2. Run container on EC2. If you need to mount local model and data into the container use `-v`:

```bash
# example: host has /home/ubuntu/model_lr and /home/ubuntu/ValidationDataset.csv
sudo docker run --rm -v /home/ubuntu/model_lr:/app/model_lr -v /home/ubuntu/ValidationDataset.csv:/app/ValidationDataset.csv \
	dockerhub_username/wine-predict:latest
```

Or pass custom args (if ENTRYPOINT accepts args):

```bash
sudo docker run --rm -v /host/model_lr:/app/model_lr -v /host/ValidationDataset.csv:/app/ValidationDataset.csv \
	dockerhub_username/wine-predict:latest model_lr ValidationDataset.csv
```


