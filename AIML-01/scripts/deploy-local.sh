#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

context="$(kubectl config current-context)"
case "$context" in
  k3d-fraud-platform|minikube) ;;
  *) echo "Select the k3d-fraud-platform or minikube context first." >&2; exit 1 ;;
esac

for file in data/processed/train.parquet artifacts/xgb_model.joblib artifacts/feature_columns.json \
    artifacts/fraud_ring_gnn.npz feast/data/registry.db feast/data/online_store.db; do
  if [[ ! -s "$file" ]]; then
    echo "Missing $file. Prepare, train, and materialize the artifacts before deploying." >&2
    exit 1
  fi
done

docker build --target service -t fraud-detection-api:local .
docker build --target dashboard -t fraud-dashboard:local .
if [[ "$context" == "minikube" ]]; then
  minikube image load fraud-detection-api:local
  minikube image load fraud-dashboard:local
else
  k3d_bin=k3d
  if [[ -x .runtime/bin/k3d ]]; then k3d_bin=.runtime/bin/k3d; fi
  "$k3d_bin" image import --mode direct -c fraud-platform fraud-detection-api:local fraud-dashboard:local
fi

kubectl --context "$context" apply -k k8s
kube=(kubectl --context "$context" -n fraud-platform)
seed="$("${kube[@]}" create -f - -o name <<'YAML'
apiVersion: v1
kind: Pod
metadata:
  generateName: fraud-training-seed-
spec:
  restartPolicy: Never
  containers:
    - name: seed
      image: busybox:1.37
      command: [sleep, "600"]
      volumeMounts:
        - {name: training, mountPath: /training}
  volumes:
    - name: training
      persistentVolumeClaim:
        claimName: fraud-training
YAML
)"
trap '"${kube[@]}" delete "$seed" --wait=false >/dev/null 2>&1 || true' EXIT
"${kube[@]}" wait --for=condition=Ready "$seed" --timeout=120s
"${kube[@]}" cp data/processed/train.parquet "${seed#pod/}:/training/train.parquet.part"
"${kube[@]}" exec "$seed" -- mv /training/train.parquet.part /training/train.parquet

for app in fraud-api fraud-dashboard; do
  "${kube[@]}" rollout restart "deployment/$app"
  "${kube[@]}" rollout status "deployment/$app" --timeout=180s
done
"${kube[@]}" rollout status statefulset/fraud-kafka --timeout=180s
"${kube[@]}" wait --for=condition=Complete job/fraud-replay --timeout=240s
echo "Open the dashboard with kubectl -n fraud-platform port-forward svc/fraud-dashboard 8080:8080"
