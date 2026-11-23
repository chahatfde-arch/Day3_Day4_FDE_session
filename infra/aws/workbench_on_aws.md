# Host the Credit Workbench on AWS

This page takes the Credit Workbench from your laptop to a server on Amazon Web Services (AWS) that you can open
in a browser. It is the **Deploy** step of Build → Deploy → Evaluate → Maintain, done for real.

What you will do:

1. Build the workbench into a container image on your laptop.
2. Push the image to **Amazon ECR**, AWS's private registry for container images.
3. Start a small server on **Amazon EC2** that pulls the image and runs it.
4. Open the workbench from your browser.
5. Delete everything, so nothing keeps costing money.

Plan about an hour the first time. Most of it is waiting for uploads and for the server to start.

Everything in the image is Good Bank's synthetic data. No API key goes into it, and the workbench runs offline on the
server, exactly as on your laptop.

---

## Before you start: what "free" means on AWS

AWS changed its free offer for new accounts on 15 July 2025. Read the current terms yourself on the
[AWS Free Tier page](https://aws.amazon.com/free/) and in the [Free Tier FAQs](https://aws.amazon.com/free/free-tier-faqs/),
because they change. What matters for this page:

- **An account created on or after 15 July 2025** chooses a **Free plan** or a Paid plan at sign-up. On the Free
  plan you get AWS credits, and the account is not charged until you choose to upgrade. The Free plan ends after
  six months or when the credits run out, whichever comes first. The server size this page uses, `t3.small`, is one
  of the EC2 sizes the Free plan allows. The server, its disk, its public address and the image storage in ECR are
  paid from the credits.
- **An account created before 15 July 2025** has the older free tier. It covers `t2.micro` or `t3.micro` servers
  for the first twelve months, and **not** `t3.small`. On such an account this page costs money for every hour the
  server exists. The amount is small for an hour or two, but it is not zero: check the
  [EC2 pricing page](https://aws.amazon.com/ec2/pricing/on-demand/) for your region, and delete everything the same day.
- **ECR storage.** The workbench image is bigger than the small ECR storage allowance of the older free tier, so
  storage beyond it is charged (or paid from credits). See the [ECR pricing page](https://aws.amazon.com/ecr/pricing/).
  Step 7 deletes the image.

Why `t3.small` and not the smaller `t3.micro`: the workbench keeps the agent and its data in memory. In the container,
during a credit review, it peaked at about 880 MB in the trainer's test. A `t3.micro` has 1 GB in total, shared with
the operating system and Docker, which is too tight; a `t3.small` has 2 GB.

**Whatever account you have, do step 1 of [README.md](README.md) first: the budget alarm.** It is what warns you if
anything is charged.

---

## What you need

| | How to check |
|---|---|
| An AWS account, with the budget alarm from [README.md](README.md) step 1 | you received the alarm's confirmation email and clicked it |
| The deploy user from [README.md](README.md) step 2, with [policy.json](policy.json) attached | `aws sts get-caller-identity` prints an ARN ending in `user/creditlens-deployer` |
| The AWS command line (AWS CLI v2) | `aws --version` prints `aws-cli/2...`. Install: [AWS CLI install guide](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) |
| Docker Desktop, running | `docker version` prints a Client and a Server |
| This repository, set up as in the main [README](../../README.md) | the workbench runs on your laptop |

**Which terminal.** Every command below is written for **bash**. On a Mac or Linux, use the Terminal. On Windows, use
**Git Bash** (it came with Git: Start menu, type *Git Bash*), not PowerShell. Run every command from the repository
folder (`cd` into it first).

**Never paste your AWS keys into a notebook, a chat or a file in this repository.** `aws configure` stores them in
your home folder, outside the repository.

---

## Step 1. Set your names once

Pick a region close to you. `ap-south-1` is Mumbai. Then set these in the terminal you will use for every step. If
you close the terminal, run this block again.

```bash
export AWS_REGION=ap-south-1
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR_REPO=creditlens-workbench
export REGISTRY=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
echo "account $ACCOUNT_ID, region $AWS_REGION, registry $REGISTRY"
```

You should see your twelve-digit account number. If you see an error instead, `aws configure` was not done (see
[README.md](README.md) step 2).

## Step 2. Build the image and try it on your laptop

```bash
docker build -f infra/Dockerfile.workbench -t creditlens-workbench:latest .
```

The first build takes several minutes: it installs the Python packages and generates Good Bank's data inside the
image. On an Apple Silicon Mac, add `--platform linux/amd64` after `docker build`, because the AWS server uses an
Intel or AMD processor.

Try it before you upload it. If the workbench is already running on your laptop, stop it first (Ctrl+C), because
both use port 8090.

```bash
docker run --rm -p 8090:8090 creditlens-workbench:latest
```

Open http://localhost:8090. Wait until the top bar says *Review assistant online* (in a container this takes about
three and a half minutes), open A901 and
click **Run credit review**: the recommendation is *Counter-offer · ₹1.50 Cr*. Then press Ctrl+C in the terminal to
stop it.

## Step 3. Push the image to ECR

```bash
# 3a. Create the repository, tagged for this project, with a security scan of every image you push.
aws ecr create-repository --region $AWS_REGION \
  --repository-name $ECR_REPO \
  --image-scanning-configuration scanOnPush=true \
  --tags Key=project,Value=creditlens

# 3b. Log Docker in to your registry. The password is a short-lived token, not your AWS secret.
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $REGISTRY

# 3c. Give the image its ECR name and upload it.
docker tag creditlens-workbench:latest $REGISTRY/$ECR_REPO:latest
docker push $REGISTRY/$ECR_REPO:latest
```

`docker login` prints *Login Succeeded*. The push uploads the image in layers; it takes a while the first time.
Check that it arrived:

```bash
aws ecr describe-images --region $AWS_REGION --repository-name $ECR_REPO \
  --query 'imageDetails[].{tags:imageTags,pushed:imagePushedAt}' --output table
```

## Step 4. A role that lets the server pull the image (once, in the console)

The server needs permission to download your image from ECR. You give that permission to the *server*, through a
role, so no key is ever stored on it. Your deploy user is not allowed to create roles (that is on purpose), so you
do this once as the account owner in the AWS console:

1. Open **IAM** → **Roles** → **Create role**.
2. **Trusted entity type**: *AWS service*. **Use case**: *EC2*. Click **Next**.
3. Search for and tick **AmazonEC2ContainerRegistryReadOnly**. Click **Next**.
4. **Role name**: `creditlens-ec2-role`. Click **Create role**.

Choosing *EC2* as the use case also creates an *instance profile* with the same name, which is how a server wears a
role. The deploy user's policy allows it to hand over only roles whose names start with `creditlens-`.

## Step 5. Start the server

```bash
# 5a. A firewall (a "security group") that lets only YOUR computer reach port 8090.
MY_IP=$(curl -s https://checkip.amazonaws.com)
SG_ID=$(aws ec2 create-security-group --region $AWS_REGION \
  --group-name creditlens-workbench-sg \
  --description "Credit Workbench, port 8090 from one address" \
  --tag-specifications 'ResourceType=security-group,Tags=[{Key=project,Value=creditlens}]' \
  --query GroupId --output text)
aws ec2 authorize-security-group-ingress --region $AWS_REGION --group-id $SG_ID \
  --protocol tcp --port 8090 --cidr $MY_IP/32
echo "security group $SG_ID allows $MY_IP"
```

Why only your address: the workbench has a demo sign-in where anyone can pick *Head of Credit* from a menu. That is
fine for synthetic data on your own screen, and wrong on the open internet.

```bash
# 5b. What the server does when it starts: install Docker, log in to ECR with its role, pull and run the image.
#     The file goes next to the repository folder (../), not inside it, so it can never be committed.
cat > ../creditlens-user-data.sh <<USERDATA
#!/bin/bash
dnf install -y docker
systemctl enable --now docker
aws ecr get-login-password --region $AWS_REGION \
  | docker login --username AWS --password-stdin $REGISTRY
docker run -d --restart unless-stopped -p 8090:8090 --name creditlens-workbench \
  $REGISTRY/$ECR_REPO:latest
USERDATA
cat ../creditlens-user-data.sh
```

Read the printed script: your registry address should appear in it, in place of the `$` names.

```bash
# 5c. The latest Amazon Linux image, then the server itself: 2 GB of memory (t3.small), a 20 GB disk (the default
#     8 GB is tight for downloading and unpacking the image), tagged for this project (the server and its disk).
AMI_ID=$(aws ssm get-parameters --region $AWS_REGION \
  --names /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64 \
  --query 'Parameters[0].Value' --output text)

INSTANCE_ID=$(aws ec2 run-instances --region $AWS_REGION \
  --image-id $AMI_ID --instance-type t3.small \
  --security-group-ids $SG_ID \
  --iam-instance-profile Name=creditlens-ec2-role \
  --block-device-mappings 'DeviceName=/dev/xvda,Ebs={VolumeSize=20,VolumeType=gp3,DeleteOnTermination=true}' \
  --user-data file://../creditlens-user-data.sh \
  --tag-specifications \
    'ResourceType=instance,Tags=[{Key=project,Value=creditlens},{Key=Name,Value=creditlens-workbench}]' \
    'ResourceType=volume,Tags=[{Key=project,Value=creditlens}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "server $INSTANCE_ID starting"

aws ec2 wait instance-running --region $AWS_REGION --instance-ids $INSTANCE_ID
aws ec2 describe-instances --region $AWS_REGION --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text
```

The last command prints the server's address, something like `ec2-13-...compute.amazonaws.com`.

## Step 6. Open it

Wait about five minutes: the server installs Docker, downloads your image and starts the workbench. The page opens as
soon as the web server is up; the top bar says *Review assistant starting* for about three and a half more minutes.
Open, in your browser:

```
http://<the address from step 5>:8090
```

You should see the same workbench as on your laptop. Run A901's credit review as in step 2. The page **Connected
systems** shows the stand-ins (*Replica*, *Estimated*): there is no Oracle or SOAP service on the server, and every
answer says so.

**If the page does not load after ten minutes**, look at what the server printed while starting:

```bash
aws ec2 get-console-output --region $AWS_REGION --instance-id $INSTANCE_ID --latest --output text | tail -40
```

| What you see | What it means |
|---|---|
| The browser waits and then gives up | Your address changed since step 5a (for example, a different Wi-Fi). Run step 5a's `curl` again and add the new address with `authorize-security-group-ingress`. |
| `no basic auth credentials` or `denied` in the console output | The server has no role, or the role lacks **AmazonEC2ContainerRegistryReadOnly**. Check step 4, then delete the server (step 7) and start a new one. |
| `exec format error` | The image was built on an Apple Silicon Mac without `--platform linux/amd64`. Rebuild with it, push again, start a new server. |
| `UnauthorizedOperation` or `AccessDenied` when you run a command | The deploy user's policy does not allow that action. The message names the action. Check that you attached [policy.json](policy.json) as it is in this repository. |

## Step 7. Delete everything

Do this the same day, even on the Free plan: credits are not unlimited.

```bash
bash infra/aws/teardown.sh            # dry run: prints what it would delete
bash infra/aws/teardown.sh --yes      # deletes the server, its disk, the security group and the ECR repository
```

It finds the server, disk and security group by the tag `project=creditlens`, and deletes every ECR repository whose
name starts with `creditlens`. It keeps the IAM user, the role and your budget alarms, on purpose. Also delete the
start-up script from step 5b: `rm ../creditlens-user-data.sh`.

The next morning, open **Billing and Cost Management** in the console. It is the only honest report of whether
everything is gone.

## Stop for the night instead of deleting

```bash
aws ec2 stop-instances --region $AWS_REGION --instance-ids $INSTANCE_ID     # stops the server
aws ec2 start-instances --region $AWS_REGION --instance-ids $INSTANCE_ID    # starts it again; the address changes
```

A stopped server is not charged for running, but its disk still is (or is paid from credits). The workbench starts
again by itself, because Docker restarts it.
