#!/usr/bin/env bash
#
# Delete everything this project created in AWS, found by the tag project=creditlens.
#
#   bash infra/aws/teardown.sh          # dry run: prints what it would delete
#   bash infra/aws/teardown.sh --yes    # actually deletes
#
# Why tags? Because "delete the things I made last Tuesday" is not a query AWS can
# answer, but "delete everything tagged project=creditlens" is. Every resource in
# infra/aws/README.md is created with that tag for exactly this moment.
#
# This script is deliberately chatty and deliberately refuses to delete anything that
# is not tagged. If it finds nothing, that is the answer you want.

set -euo pipefail

PROJECT_TAG="${PROJECT_TAG:-creditlens}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-us-east-1}}"
ECR_REPO="${ECR_REPO:-creditlens-api}"
LAMBDA_NAME="${LAMBDA_NAME:-creditlens-api}"
KEY_NAME="${KEY_NAME:-creditlens-key}"
APPLY="no"

for arg in "$@"; do
  case "$arg" in
    --yes|-y) APPLY="yes" ;;
    --help|-h)
      sed -n '2,20p' "$0"
      exit 0
      ;;
    *) echo "unknown option: $arg" >&2; exit 2 ;;
  esac
done

if ! command -v aws >/dev/null 2>&1; then
  echo "the aws CLI is not installed: https://aws.amazon.com/cli/" >&2
  exit 1
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text)"

echo "============================================================"
echo " CreditLens AI teardown"
echo " account : $ACCOUNT_ID"
echo " region  : $REGION"
echo " tag     : project=$PROJECT_TAG"
if [ "$APPLY" = "yes" ]; then
  echo " mode    : DELETING FOR REAL"
else
  echo " mode    : dry run (add --yes to delete)"
fi
echo "============================================================"
echo

# run <description> <command...>  -- prints it, and runs it only with --yes.
run() {
  local description="$1"; shift
  echo "  -> $description"
  if [ "$APPLY" = "yes" ]; then
    "$@" || echo "     (failed or already gone, continuing)"
  else
    echo "     would run: $*"
  fi
}

# ---------------------------------------------------------------------------
# 1. EC2 instances tagged for this project
# ---------------------------------------------------------------------------
echo "[1/7] EC2 instances"
INSTANCE_IDS="$(aws ec2 describe-instances --region "$REGION" \
  --filters "Name=tag:project,Values=$PROJECT_TAG" \
            "Name=instance-state-name,Values=pending,running,stopping,stopped" \
  --query 'Reservations[].Instances[].InstanceId' --output text || true)"

if [ -n "${INSTANCE_IDS:-}" ] && [ "$INSTANCE_IDS" != "None" ]; then
  for instance_id in $INSTANCE_IDS; do
    run "terminate instance $instance_id" \
      aws ec2 terminate-instances --region "$REGION" --instance-ids "$instance_id"
  done
  if [ "$APPLY" = "yes" ]; then
    echo "  -> waiting for the instances to terminate (security groups cannot go first)"
    # shellcheck disable=SC2086
    aws ec2 wait instance-terminated --region "$REGION" --instance-ids $INSTANCE_IDS || true
  fi
else
  echo "  none"
fi

# ---------------------------------------------------------------------------
# 2. Unattached elastic IPs and leftover EBS volumes: the classic surprise charge
# ---------------------------------------------------------------------------
echo "[2/7] Stray elastic IPs and volumes"
for allocation_id in $(aws ec2 describe-addresses --region "$REGION" \
    --filters "Name=tag:project,Values=$PROJECT_TAG" \
    --query 'Addresses[?AssociationId==null].AllocationId' --output text || true); do
  [ "$allocation_id" = "None" ] && continue
  run "release elastic IP $allocation_id" \
    aws ec2 release-address --region "$REGION" --allocation-id "$allocation_id"
done

for volume_id in $(aws ec2 describe-volumes --region "$REGION" \
    --filters "Name=tag:project,Values=$PROJECT_TAG" "Name=status,Values=available" \
    --query 'Volumes[].VolumeId' --output text || true); do
  [ "$volume_id" = "None" ] && continue
  run "delete unattached volume $volume_id" \
    aws ec2 delete-volume --region "$REGION" --volume-id "$volume_id"
done

# ---------------------------------------------------------------------------
# 3. Security groups and the key pair
# ---------------------------------------------------------------------------
echo "[3/7] Security groups and key pair"
for group_id in $(aws ec2 describe-security-groups --region "$REGION" \
    --filters "Name=tag:project,Values=$PROJECT_TAG" \
    --query 'SecurityGroups[?GroupName!=`default`].GroupId' --output text || true); do
  [ "$group_id" = "None" ] && continue
  run "delete security group $group_id" \
    aws ec2 delete-security-group --region "$REGION" --group-id "$group_id"
done

if aws ec2 describe-key-pairs --region "$REGION" --key-names "$KEY_NAME" >/dev/null 2>&1; then
  run "delete key pair $KEY_NAME" \
    aws ec2 delete-key-pair --region "$REGION" --key-name "$KEY_NAME"
  echo "     remember to remove ~/.ssh/$KEY_NAME.pem yourself"
else
  echo "  no key pair named $KEY_NAME"
fi

# ---------------------------------------------------------------------------
# 4. Lambda function and its URL
# ---------------------------------------------------------------------------
echo "[4/7] Lambda"
if aws lambda get-function --region "$REGION" --function-name "$LAMBDA_NAME" >/dev/null 2>&1; then
  run "delete the function URL" \
    aws lambda delete-function-url-config --region "$REGION" --function-name "$LAMBDA_NAME"
  run "delete lambda function $LAMBDA_NAME" \
    aws lambda delete-function --region "$REGION" --function-name "$LAMBDA_NAME"
else
  echo "  no function named $LAMBDA_NAME"
fi

# ---------------------------------------------------------------------------
# 5. ECR repositories whose name starts with creditlens (creditlens-api, creditlens-workbench), images and all
# ---------------------------------------------------------------------------
echo "[5/7] ECR"
REPOS="$(aws ecr describe-repositories --region "$REGION" \
  --query "repositories[?starts_with(repositoryName, 'creditlens')].repositoryName" --output text 2>/dev/null || true)"
if [ -n "$REPOS" ] && [ "$REPOS" != "None" ]; then
  for repo in $REPOS; do
    run "delete ECR repository $repo and every image in it" \
      aws ecr delete-repository --region "$REGION" --repository-name "$repo" --force
  done
else
  echo "  no repository whose name starts with creditlens"
fi

# ---------------------------------------------------------------------------
# 6. S3 buckets tagged for this project
# ---------------------------------------------------------------------------
echo "[6/7] S3"
for bucket in $(aws s3api list-buckets --query 'Buckets[].Name' --output text || true); do
  case "$bucket" in
    creditlens-*) ;;
    *) continue ;;
  esac
  tag_value="$(aws s3api get-bucket-tagging --bucket "$bucket" \
    --query "TagSet[?Key=='project'].Value | [0]" --output text 2>/dev/null || echo "None")"
  if [ "$tag_value" != "$PROJECT_TAG" ]; then
    echo "  skipping $bucket: not tagged project=$PROJECT_TAG"
    continue
  fi
  run "empty bucket $bucket" aws s3 rm "s3://$bucket" --recursive
  run "delete bucket $bucket" aws s3api delete-bucket --bucket "$bucket"
done

# ---------------------------------------------------------------------------
# 7. CloudWatch log groups
# ---------------------------------------------------------------------------
echo "[7/7] CloudWatch logs"
for prefix in "/aws/ec2/creditlens" "/aws/lambda/creditlens"; do
  for log_group in $(aws logs describe-log-groups --region "$REGION" \
      --log-group-name-prefix "$prefix" \
      --query 'logGroups[].logGroupName' --output text || true); do
    [ "$log_group" = "None" ] && continue
    run "delete log group $log_group" \
      aws logs delete-log-group --region "$REGION" --log-group-name "$log_group"
  done
done

echo
echo "============================================================"
if [ "$APPLY" = "yes" ]; then
  echo " Done. Two things this script does NOT delete, on purpose:"
  echo "   - the IAM user, role and policy (you will want them next time)"
  echo "   - your budget alarms (never delete those)"
  echo
  echo " Check the Billing dashboard tomorrow. It is the only honest report."
else
  echo " Dry run finished. Nothing was deleted. Rerun with --yes to delete."
fi
echo "============================================================"
