# Deploying CreditLens AI to AWS (milestone 10)

This is written for somebody who has never opened the AWS console. Every step says what
you are doing and why. Follow it in order, and do not skip step 1.

**About cost.** This guide names no prices, because they change and they differ by
region. Before you start, open the
[AWS Free Tier page](https://aws.amazon.com/free/) and the pricing page for each
service you are about to use, and read what the free tier actually covers in *your*
region. The budget alarm in step 1 is what protects you when something is not free.

**What you are deploying.** One container: the FastAPI service from `infra/Dockerfile`,
with the SQLite database baked in or mounted. Nothing else from the compose file goes
to AWS. Langfuse stays on Langfuse Cloud's free tier or on your laptop.

**Want to host the Credit Workbench instead (the web app, with its demo data inside)?**
Follow [workbench_on_aws.md](workbench_on_aws.md). It uses steps 1 and 2 of this page,
then gives every command for the workbench.

---

## Step 1. A budget alarm, before anything else

Do this first, every time, in every cloud account you ever touch. An alarm you set
after the bill arrives is not an alarm.

1. Sign in as the account root user, open **Billing and Cost Management**.
2. **Billing preferences** → turn on **Receive Free Tier Alerts** and enter your email.
3. **Budgets** → **Create budget** → **Use a template** → **Zero spend budget**.
   It notifies you the moment the account is charged anything at all.
4. Create a second budget, **Monthly cost budget**, with an amount you are genuinely
   willing to lose, and alerts at 50%, 80% and 100% of it.
5. Confirm the subscription email that arrives. An unconfirmed alert never fires.

Only now continue.

## Step 2. An IAM user, not the root user

The root user can close the account and change the payment method. Never use it for
day-to-day work, and turn on MFA for it today.

1. **IAM** → **Users** → **Create user**, name it `creditlens-deployer`.
2. Do not give it console access. You only need programmatic access.
3. **Permissions** → **Attach policies directly** → **Create policy** → **JSON**, and
   paste [`policy.json`](policy.json) from this folder. Name it
   `creditlens-deploy-policy`.
4. Read what that policy allows before you attach it. It can create, push to and delete
   ECR repositories whose names start with `creditlens`, launch servers and create
   security groups only when they carry the tag `project=creditlens`, change or delete
   only resources with that tag, use S3 buckets named `creditlens-*`, and write logs to
   `/aws/*/creditlens*`. It explicitly *denies* terminating a server or deleting a
   Lambda function that is not tagged `project=creditlens`. That is what "least
   privilege" means in practice: the blast radius of a mistake is one project.
5. Create an access key for the user, choose **Command Line Interface**, and store the
   two values somewhere safe. They are shown once.

```bash
aws configure            # paste the key id, the secret, your region, "json"
aws sts get-caller-identity    # should print the creditlens-deployer ARN
```

Set these once so the rest of the commands copy and paste cleanly:

```bash
export AWS_REGION=ap-south-1                      # or whichever region you chose
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=creditlens-api
export REGISTRY=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
```

## Step 3. Tag everything

Every resource you create below carries `project=creditlens`. This is not tidiness: it
is how `teardown.sh` finds things to delete, and how the deny rule in the IAM policy
knows what you are allowed to destroy. A resource you forget to tag is a resource you
will pay for and will not find.

## Step 4. Push the image to ECR

ECR is AWS's private Docker registry. Both deployment options below pull from it.

```bash
# 4a. Create the repository, with a tag and an image scan on push.
aws ecr create-repository \
  --repository-name $ECR_REPO \
  --image-scanning-configuration scanOnPush=true \
  --tags Key=project,Value=creditlens

# 4b. Log Docker in to it. The password is a short-lived token, not your AWS secret.
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $REGISTRY

# 4c. Build and push. Build from the repository root: the image needs src/.
docker build -f infra/Dockerfile -t $ECR_REPO:latest .
docker tag $ECR_REPO:latest $REGISTRY/$ECR_REPO:latest
docker push $REGISTRY/$ECR_REPO:latest
```

If you are on an Apple Silicon Mac, add `--platform linux/amd64` to `docker build`, or
the image will not run on an x86 instance.

Keep one tag. Storage beyond the free allowance is charged, so delete old images:

```bash
aws ecr list-images --repository-name $ECR_REPO
```

---

## Option A. EC2, a small virtual machine

Choose this if you want to see the container running and SSH into the box. It is the
easier mental model, and it is also the option that keeps costing money if you forget
it, because an instance is billed while it exists, whether or not anyone calls it.

Pick `t2.micro` or `t3.micro`; those are the sizes the free tier covers in most
regions. Check the free tier page for yours.

```bash
# A1. A key pair, so you can SSH in.
aws ec2 create-key-pair --key-name creditlens-key \
  --query KeyMaterial --output text > ~/.ssh/creditlens-key.pem
chmod 400 ~/.ssh/creditlens-key.pem

# A2. A security group: port 22 from your IP only, port 8000 for the API.
MY_IP=$(curl -s https://checkip.amazonaws.com)
SG_ID=$(aws ec2 create-security-group \
  --group-name creditlens-sg --description "CreditLens AI API" \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=project,Value=creditlens}]' \
  --query GroupId --output text)
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 22 --cidr $MY_IP/32
aws ec2 authorize-security-group-ingress --group-id $SG_ID \
  --protocol tcp --port 8000 --cidr 0.0.0.0/0
```

Opening 8000 to the whole internet is fine for a demo of synthetic data and wrong for
anything else. If you open it, set `API_DEV_AUTH=0` so the `X-User-Id` shortcut is off.

```bash
# A3. Launch Amazon Linux 2023, install Docker, pull and run the image.
cat > /tmp/user-data.sh <<'USERDATA'
#!/bin/bash
dnf update -y
dnf install -y docker
systemctl enable --now docker
REGION=$(curl -s http://169.254.169.254/latest/meta-data/placement/region)
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
aws ecr get-login-password --region $REGION \
  | docker login --username AWS --password-stdin $ACCOUNT.dkr.ecr.$REGION.amazonaws.com
docker run -d --restart unless-stopped -p 8000:8000 \
  --name creditlens-api \
  --log-driver=awslogs \
  --log-opt awslogs-region=$REGION \
  --log-opt awslogs-group=/aws/ec2/creditlens-api \
  --log-opt awslogs-create-group=true \
  -e LLM_OFFLINE=1 -e API_DEV_AUTH=0 \
  $ACCOUNT.dkr.ecr.$REGION.amazonaws.com/creditlens-api:latest
USERDATA

AMI_ID=$(aws ssm get-parameters \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)

aws ec2 run-instances \
  --image-id $AMI_ID --instance-type t3.micro \
  --key-name creditlens-key --security-group-ids $SG_ID \
  --iam-instance-profile Name=creditlens-ec2-profile \
  --user-data file:///tmp/user-data.sh \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=project,Value=creditlens},{Key=Name,Value=creditlens-api}]'
```

`creditlens-ec2-profile` is an instance profile wrapping a role named
`creditlens-ec2-role` that has `AmazonEC2ContainerRegistryReadOnly` and
`CloudWatchAgentServerPolicy`. Create it once in the IAM console; the instance uses it
to pull from ECR without any key on disk.

Find the address and test it:

```bash
aws ec2 describe-instances \
  --filters "Name=tag:project,Values=creditlens" "Name=instance-state-name,Values=running" \
  --query 'Reservations[].Instances[].PublicDnsName' --output text

curl http://<that-address>:8000/health
```

## Option B. Lambda from a container image, behind a function URL

Choose this if you want the thing to cost nothing while nobody is calling it. Lambda
bills per request and per millisecond of execution, so an idle deployment is idle
spend. The trade-offs: a cold start on the first request after a quiet period, a
read-only filesystem except `/tmp`, and a hard request timeout.

Add an adapter to the image. Lambda speaks its own event format, and `mangum`
translates it to ASGI:

```dockerfile
# A Lambda variant of infra/Dockerfile: same builder stage, different entrypoint.
RUN pip install mangum awslambdaric
# handler.py:  from mangum import Mangum
#              from creditlens.api.app import create_app
#              handler = Mangum(create_app())
CMD ["handler.handler"]
```

```bash
aws lambda create-function \
  --function-name creditlens-api \
  --package-type Image \
  --code ImageUri=$REGISTRY/$ECR_REPO:lambda \
  --role arn:aws:iam::$ACCOUNT_ID:role/creditlens-lambda-role \
  --timeout 30 --memory-size 1024 \
  --environment "Variables={LLM_OFFLINE=1,API_DEV_AUTH=0}" \
  --tags project=creditlens

aws lambda create-function-url-config \
  --function-name creditlens-api --auth-type AWS_IAM
```

`creditlens-lambda-role` needs `AWSLambdaBasicExecutionRole` (that is what lets it
write logs) plus ECR read. Use `--auth-type AWS_IAM` so the URL is not open to the
world; use `NONE` only for a demo you tear down the same day.

SQLite on Lambda is read-only outside `/tmp`, so bake `creditlens.db` into the image
and treat the deployment as read-only, or move to a real database. That limitation is
the lesson: serverless is cheap because it is stateless.

## Step 5. An S3 bucket with a lifecycle rule

Somewhere to put exported memos, evaluation results and drift reports. The lifecycle
rule is the cost control: objects delete themselves instead of accumulating forever.

```bash
export BUCKET=creditlens-$ACCOUNT_ID-artifacts

aws s3api create-bucket --bucket $BUCKET \
  --region $AWS_REGION \
  --create-bucket-configuration LocationConstraint=$AWS_REGION
# us-east-1 is the exception: it takes no LocationConstraint.

aws s3api put-public-access-block --bucket $BUCKET \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

aws s3api put-bucket-tagging --bucket $BUCKET \
  --tagging 'TagSet=[{Key=project,Value=creditlens}]'

cat > /tmp/lifecycle.json <<'LIFECYCLE'
{
  "Rules": [
    {
      "ID": "expire-workshop-artifacts",
      "Status": "Enabled",
      "Filter": { "Prefix": "" },
      "Transitions": [
        { "Days": 30, "StorageClass": "STANDARD_IA" }
      ],
      "Expiration": { "Days": 90 },
      "AbortIncompleteMultipartUpload": { "DaysAfterInitiation": 7 }
    }
  ]
}
LIFECYCLE

aws s3api put-bucket-lifecycle-configuration \
  --bucket $BUCKET --lifecycle-configuration file:///tmp/lifecycle.json
```

## Step 6. CloudWatch logs

The EC2 `docker run` above already ships container logs to
`/aws/ec2/creditlens-api`; Lambda writes to `/aws/lambda/creditlens-api`
automatically. Logs are stored forever by default, and storage is charged, so set a
retention period:

```bash
aws logs put-retention-policy \
  --log-group-name /aws/ec2/creditlens-api --retention-in-days 7

aws logs tail /aws/ec2/creditlens-api --follow
```

CloudWatch tells you the request happened. Langfuse tells you what the model saw and
what it answered. You want both, and they are not substitutes. Set
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` and `LANGFUSE_HOST` in the container
environment to get traces off the instance.

## Step 7. Tear it down

```bash
bash infra/aws/teardown.sh            # shows what it would delete
bash infra/aws/teardown.sh --yes      # actually deletes it
```

Then check the console the next morning. The Billing dashboard is the only honest
report of whether you really deleted everything.

## If something goes wrong

| Symptom | Usual cause |
| --- | --- |
| `denied: User is not authorized to perform: ecr:InitiateLayerUpload` | The repository name does not start with `creditlens`, so `policy.json` does not cover it. |
| `exec format error` in the container logs | Built on Apple Silicon without `--platform linux/amd64`. |
| `curl` to port 8000 hangs | The security group rule, or the instance is still running user-data. `aws ec2 get-console-output` shows the boot log. |
| Lambda returns 502 | The handler is not wrapped in `Mangum`, or the import takes longer than the timeout. |
| `/health` says `database_ok: false` | `creditlens.db` was never built into the image, or `DATABASE_URL` points somewhere that does not exist. |
| A charge appears anyway | Check for an unattached elastic IP, an EBS volume left behind by a terminated instance, or a NAT gateway you did not mean to create. |
